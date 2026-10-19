"""Combine actual per-page reviewer records after every final PNG was inspected.

Run with the bundled document Python. This tool verifies review coverage and
file identity; it never substitutes automatic checks for visual inspection.
Each input JSON identifies reviewer, status, docx_sha256, pdf_sha256, and pages
with page, png_sha256, and observation fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from pypdf import PdfReader

from core import atomic_json, digest, load_json, stamp

NAME = 'Patent_CE_十三項方法評估與專家答案比較報告'


def finalize(output_dir, reviews):
    out = Path(output_dir).resolve()
    hashes = {kind: digest(out/(NAME+'.'+kind)) for kind in ['docx', 'pdf']}
    count = len(PdfReader(str(out/(NAME+'.pdf'))).pages)
    images = {}
    for path in (out/'qa').glob('page-*.png'):
        match = re.fullmatch(r'page-(\d+)\.png', path.name)
        if match:
            number = int(match[1])
            if number in images:
                raise ValueError('Duplicate page image number')
            images[number] = path
    if count < 1 or set(images) != set(range(1, count+1)):
        raise ValueError('Rendered page coverage differs from the final PDF')
    coverage, reviewers = {}, []
    for review_path in reviews:
        record = load_json(review_path)
        if record.get('status') != 'passed' or not record.get('reviewer'):
            raise ValueError('A named reviewer must explicitly pass their pages')
        for kind, value in hashes.items():
            if record.get(kind+'_sha256') != value:
                raise ValueError('Reviewer inspected an earlier '+kind)
        reviewers.append(record['reviewer'])
        if not record.get('pages'):
            raise ValueError('Empty review record')
        for item in record['pages']:
            number = item.get('page')
            if type(number) is not int or number not in images:
                raise ValueError('Review contains an unknown page')
            if item.get('png_sha256') != digest(images[number]):
                raise ValueError('A reviewed page image changed: '+str(number))
            if not isinstance(item.get('observation'), str) or not item['observation'].strip():
                raise ValueError('Each inspected page needs an actual observation')
            coverage.setdefault(number, []).append(dict(reviewer=record['reviewer'], **item))
    if set(coverage) != set(images):
        raise ValueError('At least one final page has not been visually reviewed')
    payload = dict(status='passed', reviewed_at_utc=stamp(),
        docx_sha256=hashes['docx'], pdf_sha256=hashes['pdf'],
        page_count=count, reviewers=reviewers,
        scope='Every final rendered page inspected visually at original image resolution; Traditional Chinese glyphs, tables, charts, wrapping, and pagination checked',
        pages=[dict(page=number, image=images[number].name,
                    png_sha256=digest(images[number]), reviews=coverage[number])
               for number in sorted(images)])
    atomic_json(out/'qa_review.json', payload)
    print(f'Visual review coverage passed: {count}/{count} final pages')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    finalize(args.output_dir, args.reviews)
