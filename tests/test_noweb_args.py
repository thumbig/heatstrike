def test_noweb_import_and_main_symbol_arg():
    # simple sanity: module imports and defines main
    import noweb
    assert hasattr(noweb, "main")
