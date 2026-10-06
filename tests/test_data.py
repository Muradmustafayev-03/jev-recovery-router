from collections import Counter

from recovery_router.data import build_roles, leakage_audit, normalize_text, resolve_collisions


def synthetic_source():
    data = {k: [] for k in ["train", "val", "test", "oos_train", "oos_val", "oos_test"]}
    for split, count in [("train", 100), ("val", 20), ("test", 30)]:
        for label_id in range(150):
            label = f"intent_{label_id:03d}"
            for i in range(count):
                rid = f"clinc:{split}:{label}:{i}"
                data[split].append({"id": rid, "parent_id": rid, "text": rid, "label": label,
                                    "dataset": "clinc150", "source_split": split, "is_oos": False})
    for split, count in [("oos_train", 250), ("oos_val", 100), ("oos_test", 1000)]:
        for i in range(count):
            rid = f"clinc:{split}:{i}"
            data[split].append({"id": rid, "parent_id": rid, "text": rid, "label": "oos",
                                "dataset": "clinc150", "source_split": split, "is_oos": True})
    bank = []
    for label_id in range(77):
        for i in range(25):
            rid = f"bank:train:{label_id}:{i}"
            bank.append({"id": rid, "parent_id": rid, "text": rid, "label": f"bank_{label_id}",
                         "dataset": "banking77", "source_split": "train", "is_oos": False})
    test = [{"id": "bank:test:0", "parent_id": "bank:test:0", "text": "bank test example",
             "label": "bank_0", "dataset": "banking77", "source_split": "test", "is_oos": False}]
    return data, bank, test


def test_splits_preserve_sealed_novel_classes_and_index_separation():
    inputs = synthetic_source()
    roles, catalogs, partition = build_roles(*inputs, seed=42)
    assert len(roles["train"]) == 7400
    assert len(roles["policy_train"]) == 2450
    assert len(roles["calibration"]) == len(roles["tuning"]) == 1250
    assert len(roles["novel_validation"]) == 600
    assert len(roles["test"]) == 5500
    assert len(roles["test_base"]) == 4600
    assert len(roles["index"]) == 3000
    assert set(Counter(r["label"] for r in roles["index"]).values()) == {20}
    for role in ("train", "policy_train", "calibration", "tuning"):
        assert not set(r["label"] for r in roles[role]).intersection(partition["novel_labels"])
        assert {r["catalog_id"] for r in roles[role]} == {"clinc120"}
    assert set(catalogs["clinc120"]).isdisjoint(partition["novel_labels"])
    assert all(r["catalog_id"] == "clinc150" for r in roles["test"])
    exclusive = {k: roles[k] for k in ("index", "train", "policy_train", "calibration", "tuning", "test", "novel_validation", "banking_index", "transfer")}
    assert leakage_audit(exclusive)["cross_role_text_collisions"] == 0
    assert roles == build_roles(*inputs, seed=42)[0]


def test_audit_detects_normalized_cross_role_duplicate():
    train = {"id": "train", "text": " MY card, please! ", "label": "a"}
    test = {"id": "test", "text": "my card please", "label": "b"}
    report = leakage_audit({"train": [train], "test": [test]})
    assert report["cross_role_text_collisions"] == 1
    assert normalize_text(train["text"]) == normalize_text(test["text"])
    assert {e["label"] for e in report["collisions"][0]["entries"]} == {"a", "b"}


def test_different_seed_changes_catalog_partition():
    source = synthetic_source()
    assert build_roles(*source, seed=42)[2] != build_roles(*source, seed=43)[2]


def test_resolution_keeps_tests_and_removes_every_matching_fitting_row():
    roles, _, _ = build_roles(*synthetic_source(), seed=42)
    target = roles["test"][0]["text"]
    roles["train"][0]["text"] = target
    roles["index"][0]["text"] = target
    roles["direct_classifier_train"][0]["text"] = target
    roles["transfer"][0]["text"] = target
    original_test = list(roles["test"])
    cleaned, report = resolve_collisions(roles)
    assert cleaned["test"] == original_test
    assert cleaned["transfer"][0]["text"] == target
    assert report["evaluation_overlap"]["cross_role_text_collisions"] == 1
    assert report["remaining_fitting_collisions"] == 0
    for role in ("train", "index", "direct_classifier_train"):
        assert target not in {r["text"] for r in cleaned[role]}
