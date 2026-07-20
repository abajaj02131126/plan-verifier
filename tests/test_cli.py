from verifier.__main__ import build_parser, main


def test_parser_builds():
    parser = build_parser()
    assert parser.prog == "verifier"


def test_main_status_returns_zero():
    assert main(["status"]) == 0
