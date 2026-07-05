# Notebooks

Every notebook here mirrors code in src/ and must be executed top-to-bottom before packaging (no stripped or un-run cells). The set of notebooks must reproduce the reported results, regenerate every figure used in the manuscript, and match the outputs produced in the working environment. tools/package_repo.py enforces that notebooks are present and executed before it writes the GitHub ZIP.
