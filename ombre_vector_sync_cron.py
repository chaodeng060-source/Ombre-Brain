#!/usr/bin/env python3
"""NAS 容器内定时跑：桶文件 → sqlite embeddings → PG ombre_vectors 三者对齐。

朝灯 2026-09-09「没有噪音，召回完整」。当天查出两个病，都是这条链没人守：
  · 881 个桶（9 月 249 条抽样里占大半）embedding 算好了却从没同步进 PG，
    向量召回根本看不见它们 —— 「召回不完整」的真病。
  · 71 个已删桶的 embedding 还留在 sqlite 和 PG 里，每次召回都可能被捞上来，
    正文空、摘要残 —— 「噪音」的一部分。删桶入口只删 md/BM25，不删 embedding。

sqlite /data/embeddings.db 是 embedding 真源，PG 是它的检索副本；
桶 .md 文件是记忆本身的真源。所以对齐顺序固定：文件 → sqlite → PG。
只删「文件已经不在」的，绝不反过来拿 PG 去删文件。

用法（容器内）：
    python3 ombre_vector_sync_cron.py            # dry-run，只报数
    python3 ombre_vector_sync_cron.py --execute  # 真跑

安全闸：桶文件数为 0（挂载没起来/路径错）时直接退出，不删任何东西。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import psycopg

EXECUTE = "--execute" in sys.argv
BUCKETS_DIR = Path(os.environ.get("OMBRE_BUCKETS_DIR", "/data"))
EMBEDDINGS_DB = BUCKETS_DIR / "embeddings.db"
DSN = os.environ["OMBRE_PG_RECALL_DSN"]

_ID_RE = re.compile(r"_([0-9a-f]{12})\.md$")


def bucket_ids_on_disk() -> set[str]:
    out: set[str] = set()
    for p in BUCKETS_DIR.rglob("*.md"):
        m = _ID_RE.search(p.name)
        if m:
            out.add(m.group(1))
    return out


def main() -> int:
    disk = bucket_ids_on_disk()
    print(f"disk buckets={len(disk)}")
    if not disk:
        print("FATAL: 硬盘上一个桶都没有，挂载可能没起来；不做任何删除", file=sys.stderr)
        return 2

    sq = sqlite3.connect(str(EMBEDDINGS_DB))
    sq_ids = {r[0] for r in sq.execute("select bucket_id from embeddings")}
    sq_ghosts = sorted(sq_ids - disk)
    print(f"sqlite rows={len(sq_ids)} ghosts={len(sq_ghosts)}")

    with psycopg.connect(DSN, connect_timeout=10) as pg:
        with pg.cursor() as cur:
            cur.execute(
                "select bucket_id, max(source_updated_at) from ombre_vectors group by bucket_id"
            )
            pg_rows = dict(cur.fetchall())

        live_src = {}
        for bid, emb, upd in sq.execute(
            "select bucket_id, embedding, updated_at from embeddings"
        ):
            if bid in disk:
                live_src[bid] = (emb, upd)

        new = [b for b in live_src if b not in pg_rows]
        stale = [
            b for b in live_src
            if b in pg_rows and str(live_src[b][1]) > str(pg_rows[b])
        ]
        pg_ghosts = sorted(set(pg_rows) - disk)
        missing_embedding = sorted(disk - sq_ids)
        print(
            f"pg buckets={len(pg_rows)} new={len(new)} stale={len(stale)} "
            f"pg_ghosts={len(pg_ghosts)} no_embedding_yet={len(missing_embedding)}"
        )

        if not EXECUTE:
            print("dry-run，加 --execute 真跑")
            return 0

        if sq_ghosts:
            sq.executemany(
                "delete from embeddings where bucket_id=?", [(b,) for b in sq_ghosts]
            )
            sq.commit()
            print(f"sqlite ghosts deleted={len(sq_ghosts)}")

        if pg_ghosts:
            with pg.cursor() as cur:
                cur.execute(
                    "delete from ombre_vectors where bucket_id = any(%s::text[])",
                    (pg_ghosts,),
                )
                cur.execute(
                    "delete from ombre_bodies where bucket_id = any(%s::text[])",
                    (pg_ghosts,),
                )
            pg.commit()
            print(f"pg ghosts deleted={len(pg_ghosts)}")

        synced = 0
        skipped = 0
        with pg.cursor() as cur:
            for bid in new + stale:
                emb, upd = live_src[bid]
                vec = json.loads(emb)
                # 真源格式：分段嵌套 [[1024 维], ...]；旧平铺 [1024 维] 也兼容
                if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
                    segments = [vec]
                elif isinstance(vec, list):
                    segments = vec
                else:
                    skipped += 1
                    continue
                if any(not isinstance(s, list) or len(s) != 1024 for s in segments):
                    skipped += 1
                    continue
                cur.execute("delete from ombre_vectors where bucket_id=%s", (bid,))
                for i, seg in enumerate(segments):
                    vec_text = "[" + ",".join(repr(float(x)) for x in seg) + "]"
                    cur.execute(
                        "insert into ombre_vectors "
                        "(bucket_id, segment_idx, embedding, source_updated_at, mirrored_at) "
                        "values (%s, %s, %s::vector, %s, now())",
                        (bid, i, vec_text, str(upd)),
                    )
                synced += 1
        pg.commit()
        print(f"synced={synced} skipped_bad_dims={skipped}")

        with pg.cursor() as cur:
            cur.execute("select count(distinct bucket_id) from ombre_vectors")
            final = cur.fetchone()[0]
        print(f"post-check: disk={len(disk)} pg_buckets={final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
