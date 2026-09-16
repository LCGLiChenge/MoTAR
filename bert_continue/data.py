"""Fold original 4x112 global buckets into 2x112x2 without changing rows."""
from bert2d.runtime import RankBucketSampler


class FoldedSampler:
    def __init__(self, dataset, rank, seed, micro=112, world=2):
        if world not in (2,4) or not 0<=rank<world or micro < 1 or (448//world) % micro:
            raise ValueError("two/four ranks, micro must divide per-rank global batch")
        self.global_sampler = RankBucketSampler(dataset, 448, 1, 0, seed)
        self.rank = rank
        self.micro = micro
        self.world = world
        self.accum = 448 // world // micro

    def set_epoch(self, epoch):
        self.global_sampler.set_epoch(epoch)

    def __len__(self):
        return self.accum * len(self.global_sampler)

    def __iter__(self):
        for rows in self.global_sampler:
            for chunk in range(self.accum):
                start = (self.world * chunk + self.rank) * self.micro
                yield rows[start:start+self.micro]


def convert_cursor(meta, micro=112, world=2):
    if world not in (2,4) or micro < 1 or (448//world) % micro:
        raise ValueError("micro must divide per-rank global batch")
    new_accum=448//world//micro
    cursor = dict(meta["cursor"])
    cfg = meta["config"]
    if cfg["world"] == 4:
        if cfg["micro"] != 112 or cfg["accumulation"] != 1:
            raise ValueError("only original 4x112 checkpoint migration supported")
        cursor["next_microbatch_offset"] *= new_accum
    elif cfg["world"] != 2 or cfg.get("sampler_layout") != "folded_original_4x112":
        raise ValueError("unrecognized continuation sampler")
    else:
        old_accum=224//cfg["micro"]
        if cfg["micro"]<1 or 224%cfg["micro"] or cursor["next_microbatch_offset"]%old_accum:
            raise ValueError("source cursor is not at a complete global batch")
        cursor["next_microbatch_offset"]=cursor["next_microbatch_offset"]//old_accum*new_accum
    if cursor["next_microbatch_offset"] % new_accum:
        raise ValueError("cursor is not at a complete optimizer update")
    return cursor
