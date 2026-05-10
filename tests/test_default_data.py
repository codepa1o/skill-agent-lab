from app import default_data, run_store


def test_default_test_suite_is_seeded_once(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(default_data, "list_test_suites", run_store.list_test_suites)
    monkeypatch.setattr(default_data, "create_test_suite", run_store.create_test_suite)
    monkeypatch.setattr(default_data, "create_test_case", run_store.create_test_case)

    default_data.ensure_default_test_suites()
    default_data.ensure_default_test_suites()

    suites = run_store.list_test_suites()
    cases = run_store.list_test_cases(suites[0].id)

    assert len(suites) == 1
    assert len(cases) == len(default_data.DEFAULT_CASES)
