"""
Live session demo. Embeddings + quantized Qwen, all local.

Run:  .venv\\Scripts\\python main.py
"""

import summary_tree
from summary_tree import Tree

FAKE_TRANSCRIPT = [
    "Alright let's start the standup, quick round of updates.",
    "First up the payments migration, we moved 60 percent of traffic.",
    "Error rate stayed flat at 0.2 percent so far.",
    "We still need to migrate refunds, that's next week.",
    "On refunds specifically, the old provider api is being shut off in May.",
    "So we have a hard deadline there, no room to slip.",
    "Switching topics, the mobile release got rejected by review.",
    "They flagged the microphone permission description.",
    "We resubmitted yesterday with a longer explanation.",
    "Last thing, hiring. Two onsites this week for the backend role.",
    "Both candidates look strong, decision by Friday.",
]


def main():
    print("loading nomic from models/ ...")
    from local_models import ensure_embedder, qwen_ready

    ensure_embedder()
    has_qwen = qwen_ready()
    print("qwen ready:", has_qwen)

    tree = Tree(
        word_threshold=20 if has_qwen else 9999,
        time_threshold=9999,
        offline=False,
    )

    for i, chunk in enumerate(FAKE_TRANSCRIPT, 1):
        node = tree.insert_chunk(chunk)
        print(f"\n=== chunk {i} -> {node.title} (d{node.depth}) [{tree.last_reason}] ===")
        print(f'"{chunk}"')
        print(tree.outline())
        print(f"llm calls so far: {summary_tree.llm_call_count}")

    print("\n=== session over: flush buffers + roll up ===")
    tree.flush_all()
    rolled = tree.rollup_dirty_nodes()
    print(tree.outline())
    print(f"\nrolled up {rolled} dirty nodes")
    print(f"total llm calls: {summary_tree.llm_call_count}")
    print(f"\nwhole meeting so far:\n{tree.root.subtree_summary}")


if __name__ == "__main__":
    main()
