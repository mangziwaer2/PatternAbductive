import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hypothesis_explainer import HypothesisParseError, explain_hypothesis_text


def parse_args():
    parser = argparse.ArgumentParser(description='Explain a structured hypothesis string.')
    parser.add_argument('--text', default='', help='Hypothesis text to explain.')
    return parser.parse_args()


def main():
    args = parse_args()
    hypothesis_text = args.text.strip()
    if not hypothesis_text:
        hypothesis_text = sys.stdin.read().strip()
    if not hypothesis_text:
        raise SystemExit('Provide hypothesis text via --text or stdin.')

    try:
        explained = explain_hypothesis_text(hypothesis_text)
    except HypothesisParseError as exc:
        raise SystemExit(f'Parse error: {exc}') from exc

    print('Hypothesis:')
    print(hypothesis_text)
    print()
    print(f'Pattern: {explained["pattern"]}')
    print(f'Anchor entities: {explained["anchors"]}')
    print(f'Relations: {explained["relations"]}')
    print()
    print('Structure:')
    for line in explained['tree_lines']:
        print(line)
    print()
    print('Logic:')
    print(explained['logic_expression'])
    print()
    print('Readable gloss:')
    print(explained['gloss'])


if __name__ == '__main__':
    main()
