# Notebooks

For exploration only (looking at data, drafting charts). Rules:

- No business logic lives here. Once a notebook cell is useful, move it into `src/radar/` with a test.
- Import from the package (`from radar.parse.xbrl_csv import read_facts_zip`), never copy code.
- Clear outputs before committing (outputs may contain data you should not publish).
