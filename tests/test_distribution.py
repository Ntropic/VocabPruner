from importlib import metadata, resources

from vocabpruner.analysis.groups import selection_groups
from vocabpruner.analysis.scripts import script_lists
from vocabpruner.analysis.symbols import symbol_lists, syntax_lists
from vocabpruner.checkpoints.codebooks import codebook_rules
from vocabpruner.presentation.artifacts import write_wordfreq_html


def test_installed_distribution_exposes_cli():
    package = metadata.distribution("vocabpruner")
    entrypoints = {entry.name: entry.value for entry in package.entry_points if entry.group == "console_scripts"}
    assert entrypoints["vocabpruner"] == "vocabpruner.cli:main"


def test_installed_resources_and_viewer(tmp_path):
    assert selection_groups()["latin-all"]
    assert script_lists()
    assert symbol_lists()
    assert syntax_lists()
    assert codebook_rules()
    template = resources.files("vocabpruner.presentation").joinpath("viewer.html")
    assert template.is_file()
    html = write_wordfreq_html(tmp_path).read_text(encoding="utf-8")
    assert 'fetch("settings.json")' in html
    assert 'fetch("token_facts.json")' in html
    assert 'fetch("wordfreq_facts.json")' in html
    assert "__VOCABPRUNER_" not in html
    assert "${error}" in html
