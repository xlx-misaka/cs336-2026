import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from warcio.warcwriter import WARCWriter

from scripts import filter_data as pipeline


def write_shard(path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts), encoding="utf-8")


def test_post_dedup_length_counts_and_preserves_text(tmp_path):
    long_text = "\n    ".join(f"word{index}" for index in range(50))
    short_text = " ".join(f"short{index}" for index in range(49))
    source = tmp_path / "input" / "wet_0000.jsonl"
    write_shard(source, [long_text, short_text, "", " \n\t"])
    output = tmp_path / "output"
    stats = pipeline._filter_short_documents([source], output, 50)
    assert list(pipeline._read_documents(output / source.name)) == [long_text]
    assert stats["input_documents"] == 4
    assert stats["kept_documents"] == 1
    assert stats["empty_discarded"] == 2
    assert stats["short_discarded"] == 1
    assert stats["discarded_documents"] == 3


def test_post_dedup_zero_threshold_keeps_short_nonempty_text(tmp_path):
    source = tmp_path / "input" / "wet_0000.jsonl"
    write_shard(source, ["short\ntext", " \n"])
    stats = pipeline._filter_short_documents([source], tmp_path / "output", 0)
    assert list(pipeline._read_documents(tmp_path / "output" / source.name)) == ["short\ntext"]
    assert stats["empty_discarded"] == 1
    assert stats["short_discarded"] == 0


def test_parallel_minhash_matches_serial_with_candidate_collisions(tmp_path):
    first = "café apple birch cedar elm\n\nfir ginkgo hazel ivy juniper kapok larch maple oak pine"
    unrelated = "ocean river mountain valley canyon desert island lake meadow forest glacier waterfall beach"
    paths = [tmp_path / "input" / f"wet_{index:04d}.jsonl" for index in range(2)]
    write_shard(paths[0], [first, unrelated, "tiny", ""])
    write_shard(paths[1], [first.upper().replace("\n", " ")] * 16)
    serial = pipeline._minhash_document_deduplication(paths, tmp_path / "serial", workers=1)
    parallel = pipeline._minhash_document_deduplication(paths, tmp_path / "parallel", workers=2)
    assert serial == parallel
    assert serial["input_documents"] == 20
    assert serial["kept_documents"] == 4
    for path in paths:
        assert (tmp_path / "serial" / path.name).read_bytes() == (tmp_path / "parallel" / path.name).read_bytes()


def test_pipeline_report_and_final_only_output(tmp_path, monkeypatch):
    first = "shared navigation\n" + " ".join(f"first{index}" for index in range(60)) + "\n"
    last = "shared navigation\n" + " ".join(f"last{index}" for index in range(60)) + "\n"
    paths = [tmp_path / f"input_{index}.wet.gz" for index in range(2)]
    for path, texts in zip(paths, [[first, first], ["brief content", last]], strict=True):
        with path.open("wb") as destination:
            writer = WARCWriter(destination, gzip=True)
            for index, text in enumerate(texts):
                writer.write_record(
                    writer.create_warc_record(
                        f"https://example.com/{path.name}/{index}",
                        "conversion",
                        payload=BytesIO(text.encode()),
                        warc_headers_dict={"Content-Type": "text/plain"},
                    )
                )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"$furu": {"value": str(path)}} for path in paths]))
    monkeypatch.setattr(pipeline.concurrent.futures, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(pipeline, "gopher_quality_filter", lambda text: True)
    monkeypatch.setattr(pipeline, "classify_quality", lambda text: ("wiki", 0.99))
    monkeypatch.setattr(pipeline, "classify_nsfw", lambda text: ("non-nsfw", 0.99))
    monkeypatch.setattr(pipeline, "classify_toxic_speech", lambda text: ("non-toxic", 0.99))
    output = tmp_path / "output"
    pipeline.run(["--manifest", str(manifest), "--output-dir", str(output), "--hash-workers", "1", "--min-words", "50"])
    report = json.loads((output / "filter_stats.json").read_text())
    stages = {stage["name"]: stage for stage in report["stages"]}
    assert report["input_documents"] == 4
    assert report["final_output_documents"] == report["final_nonempty_documents"] == 2
    assert report["final_empty_documents"] == 0
    assert stages["exact_document_deduplication"]["discarded_documents"] == 1
    assert stages["post_dedup_quality"]["empty_discarded"] == 0
    assert stages["post_dedup_quality"]["short_discarded"] == 1
    assert sum(stage["discarded_documents"] for stage in report["stages"]) == report["total_discarded_documents"] == 2
    assert sorted(path.name for path in output.iterdir()) == ["filter_stats.json", "wet_0000.jsonl", "wet_0001.jsonl"]
    assert list(pipeline._read_documents(output / "wet_0001.jsonl")) == [
        last.removeprefix("shared navigation\n").strip()
    ]
