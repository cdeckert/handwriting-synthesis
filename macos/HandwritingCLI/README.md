# Handwriting CLI for macOS

`handwriting-cli` turns UTF-8 text into handwritten PDF, PNG, or SVG pages. The
neural network runs locally through Apple Core ML, and long input is wrapped and
split across multiple pages automatically. No text is sent over the network.

## Requirements

- Apple silicon Mac
- macOS 14 or newer
- Xcode 16 or newer

## Build

```bash
cd macos/HandwritingCLI
swift build -c release
```

The executable is written to `.build/release/handwriting-cli`.

## Examples

```bash
.build/release/handwriting-cli \
  --text "Ein kurzer handschriftlicher Text." \
  --output note.pdf

.build/release/handwriting-cli \
  --input brief.txt \
  --output brief.pdf \
  --style 4 \
  --font-size 32 \
  --seed 2026

cat brief.txt | .build/release/handwriting-cli \
  --input - \
  --output seite.png
```

PDF output is one multi-page file. PNG and SVG output use numbered filenames
such as `seite-001.png`, `seite-002.png`, and so on. Successful runs print a
machine-readable JSON result to stdout; progress and errors go to stderr.

Use `--list-styles` to see the 13 bundled handwriting styles and `--help` for
all options.

## Character support

The default `--engine auto` mode preserves the input exactly. Text supported by
the original neural model uses neural stroke synthesis. If the document contains
German umlauts, `ß`, accents, typographic punctuation, currency symbols, or
characters from another script, the CLI automatically uses the local Unicode
handwriting renderer for the complete document. It uses Apple's Noteworthy font
by default and Core Text font fallback for glyphs that Noteworthy does not
contain.

You can select the renderer explicitly:

```bash
# Exact Unicode support with an Apple handwriting font
.build/release/handwriting-cli --engine unicode --font Noteworthy-Light \
  --text "Ärger, Öl, Grüße, 25 € – déjà vu!" --output unicode.pdf

# Original recurrent neural model; rejects characters outside its alphabet
.build/release/handwriting-cli --engine neural \
  --text "Neural handwriting" --output neural.pdf
```
