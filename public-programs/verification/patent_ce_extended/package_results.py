"""Package only reviewed extended results and their immutable reproduction inputs."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

from core import (METHODS, NEW_METHODS, PROJECT, LEGACY, atomic_json, digest,
                  load_json, stamp, wsl_path)

REPORT_NAME = 'Patent_CE_十三項方法評估與專家答案比較報告'
ARCHIVE_NAME = 'Patent_CE_十三項方法完整評估與重跑資料.zip'
REQUIRED_RESULTS = [
    'results.json', 'scores.csv', 'predictions.csv', 'summary.csv', 'comparisons.json',
    'pairs.json', 'pair_metadata.csv', 'pair_metadata.json', 'data_dictionary.json',
    'verification.json', 'qa_review.json', 'report_build_manifest.json',
    'legacy_import_verification.json', 'assets.json', 'extended_environment_lock.json',
    'moverscore_v2_idf.json',
    'alignscore_optimization_validation.json',
    'supplemental_sample_validation.json',
    REPORT_NAME+'.docx', REPORT_NAME+'.pdf',
]
OPTIONAL_RESULTS = ['report_content.md', 'artifact_operation_markers.json',
                    'numerical_tests.json', 'tests_verification.json',
                    'source_environment_lock.json', 'document_environment_lock.json',
                    'assets_resume_verification.json', 'bart_shared_embedding_audit.json',
                    'align_expansion_inventory.json', 'align_fallback_tokenization_audit.json']


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Required publication artifact is missing: '+str(path))
    return path.resolve()


def project_relative(path):
    path = Path(path).resolve()
    try:
        relative = path.relative_to(PROJECT.resolve())
    except ValueError as error:
        raise ValueError('Publication artifacts must remain within the named project root: '+str(path)) from error
    if path.is_symlink() or any(part in ['__pycache__', '.venv', 'site-packages'] for part in relative.parts):
        raise ValueError('Runtime/cache content is excluded from the publication set')
    return relative.as_posix()


def supplemental_files(out):
    from supplemental_validation import select_samples, compare_scores, parse_resources, w
    prepared = load_json(out/'pairs.json')
    selected, selection = select_samples(prepared)
    proof = load_json(require_file(out/'supplemental_sample_validation.json'))
    subdir = out/'supplemental_sample_validation'
    subset_path = require_file(subdir/'pairs.json')
    subset = load_json(subset_path)
    if proof.get('status') != 'passed' or proof.get('complete') is not True or proof.get('selection') != selection or set(proof.get('methods', {})) != set(NEW_METHODS):
        raise ValueError('Shortest/median official validation must pass for all eight new methods')
    for actual, expected in [
        (proof.get('original_input_sha256'), digest(out/'pairs.json')),
        (proof.get('subset_input_sha256'), digest(subset_path)),
        (proof.get('original_pairs_sha256'), w.value_hash(prepared['pairs'])),
        (proof.get('subset_pairs_sha256'), w.value_hash(selected)),
        (proof.get('retained_records_sha256'), w.value_hash(prepared['records'])),
        (proof.get('training_records_sha256'), w.value_hash([r for r in prepared['records'] if r.get('split')=='train'])),
        (proof.get('worker_sha256'), digest(Path(__file__).parent/'worker.py')),
        (proof.get('helper_sha256'), digest(Path(__file__).parent/'supplemental_validation.py'))]:
        if actual != expected:
            raise ValueError('Supplemental validation has a stale source signature')
    if subset.get('records') != prepared['records'] or subset.get('pairs') != selected or proof.get('training_records_retained_exactly') is not True:
        raise ValueError('Supplemental validation altered raw samples or training IDF population')
    paths = {subset_path}
    ids = {p['pair_id'] for p in selected}
    for method in NEW_METHODS:
        directory = subdir/'smoke'/method
        score_path = require_file(directory/(method+'_scores.json'))
        score = load_json(score_path)
        formal = load_json(out/(method+'_scores.json'))
        record = proof['methods'][method]
        if score.get('input_sha256') != digest(subset_path) or score.get('complete') is not True or score.get('smoke') is not True:
            raise ValueError('Supplemental score output is incomplete or has a different input')
        comparison = compare_scores(method, formal, score, ids)
        if any(record.get(k) != v for k, v in comparison.items()) or record.get('result_sha256') != digest(score_path):
            raise ValueError('Supplemental official/formal comparison record is stale')
        if record.get('formal_result_sha256') != digest(out/(method+'_scores.json')) or record.get('formal_resume_proof_sha256') != digest(out/(method+'_resume_verification.json')):
            raise ValueError('Formal results changed after supplemental validation')
        resources = record.get('resources', {})
        measurements = resources.get('measurements', [])
        if not measurements or measurements[0].get('completed_output_existed_before_invocation') is not False:
            raise ValueError('The initial supplemental model process must have actual measured resources')
        context = dict(method=method, worker_sha256=score['worker_sha256'], input_sha256=digest(subset_path), assets_sha256=score['assets_sha256'], environment_sha256=score['environment_sha256'])
        for measured in measurements:
            filename = PurePosixPath(measured['raw_path']).name
            if not re.fullmatch(re.escape(method)+r'_resources_\d+_\d+\.txt', filename) or measured['raw_path'] != wsl_path(subdir/filename):
                raise ValueError('Resource evidence must remain in the named supplemental directory')
            raw_path = require_file(subdir/filename)
            actual = parse_resources(raw_path)
            keys = ['peak_rss_kbytes', 'wall_seconds', 'user_cpu_seconds', 'system_cpu_seconds', 'exit_status', 'raw_sha256']
            if measured.get('context') != context or any(measured.get(k) != actual[k] for k in keys):
                raise ValueError('Supplemental resource evidence differs from the actual GNU time output')
            paths.add(raw_path)
        if resources.get('peak_rss_kbytes') != max(x['peak_rss_kbytes'] for x in measurements) or resources.get('initial_peak_rss_kbytes') != measurements[0]['peak_rss_kbytes'] or resources.get('initial_wall_seconds') != measurements[0]['wall_seconds']:
            raise ValueError('Supplemental resource summary lost the original model-process measurement')
        resume_path = require_file(directory/(method+'_resume_verification.json'))
        resume = load_json(resume_path)
        if resume != record.get('zero_inference_resume') or record.get('resume_proof_sha256') != digest(resume_path) or resume.get('status') != 'passed' or resume.get('inference_calls') != 0 or resume.get('original_output_sha256') != digest(score_path):
            raise ValueError('Supplemental no-inference resume proof is stale')
        paths.update([score_path, resume_path])
        for suffix in ['state.json', 'checkpoint.jsonl']:
            paths.add(require_file(directory/(method+'_'+suffix)))
        if method == 'moverscore_v2':
            idf_path = require_file(directory/'moverscore_v2_idf.json')
            if load_json(idf_path) != load_json(out/'moverscore_v2_idf.json') or digest(idf_path) != record.get('idf', {}).get('supplemental_file_sha256'):
                raise ValueError('Supplemental MoverScore IDF differs from the formal training-only IDF')
            paths.add(idf_path)
        if method == 'alignscore':
            parity_path = require_file(directory/'alignscore_optimization_parity.json')
            if digest(parity_path) != record.get('align_optimization_proof', {}).get('sha256') or load_json(parity_path).get('status') != 'passed':
                raise ValueError('Supplemental AlignScore optimization parity is stale')
            paths.add(parity_path)
    return paths


def select_files(output_dir):
    out = Path(output_dir).resolve()
    result = load_json(require_file(out/'results.json'))
    verification = load_json(require_file(out/'verification.json'))
    qa = load_json(require_file(out/'qa_review.json'))
    if verification.get('status') != 'passed' or result.get('verification') != verification:
        raise ValueError('Current independent numerical verification must pass before packaging')
    if qa.get('status') != 'passed':
        raise ValueError('Every final document page must pass visual review before packaging')
    page_count = qa.get('page_count')
    pages = qa.get('pages', [])
    if type(page_count) is not int or page_count < 1 or len(pages) != page_count or {item.get('page') for item in pages} != set(range(1, page_count+1)):
        raise ValueError('Visual review must cover every final page exactly once')
    for item in pages:
        image_path = require_file(out/'qa'/item['image'])
        if digest(image_path) != item.get('png_sha256') or not item.get('reviews'):
            raise ValueError('A final page image changed or lacks a visual review')
    optimization = load_json(require_file(out/'alignscore_optimization_validation.json'))
    align = load_json(require_file(out/'alignscore_scores.json'))
    if optimization.get('status') != 'passed' or optimization.get('native_forward_unmodified') is not True or optimization.get('native_reference_bypasses_cache') is not True:
        raise ValueError('The AlignScore inference optimization must pass an unmodified-reference comparison')
    for key in ['worker_sha256','input_sha256','assets_sha256','environment_sha256']:
        if optimization.get(key) != align.get(key):
            raise ValueError('AlignScore optimization evidence has a stale signature: '+key)
    for suffix in ['docx', 'pdf']:
        document = require_file(out/(REPORT_NAME+'.'+suffix))
        if digest(document) != qa.get(suffix+'_sha256'):
            raise ValueError('Final '+suffix+' changed after visual review')
    build = load_json(require_file(out/'report_build_manifest.json'))
    if build.get('builder_sha256') != digest(Path(__file__).parent/'build_report.py'):
        raise ValueError('The report builder changed after final authoring')
    for name, field in [('results.json', 'source_results_sha256'), ('verification.json', 'source_verification_sha256')]:
        if digest(out/name) != build.get(field):
            raise ValueError('Report was built from an earlier '+name)
    if build.get('docx_sha256') != digest(out/(REPORT_NAME+'.docx')):
        raise ValueError('Report build manifest does not identify the final DOCX')
    if not build.get('figures'):
        raise ValueError('The report build must identify its final figures')
    for name, expected in build['figures'].items():
        if digest(require_file(out/'figures'/name)) != expected:
            raise ValueError('Report figure changed after authoring: '+name)
    for name, expected in verification.get('artifact_hashes', {}).items():
        if digest(require_file(out/name)) != expected:
            raise ValueError('Verified numerical artifact changed: '+name)

    paths = {require_file(out/name) for name in REQUIRED_RESULTS}
    paths.update((out/name).resolve() for name in OPTIONAL_RESULTS if (out/name).is_file())
    input_hash = digest(out/'pairs.json')
    prepared_pairs = load_json(out/'pairs.json')['pairs']
    expected_ids = {item['pair_id'] for item in prepared_pairs}
    selected_smoke = []
    for same in [True, False]:
        found = next((item for item in prepared_pairs if (item['gold_claim'] == item['candidate']) is same), None)
        if found is not None and found not in selected_smoke:
            selected_smoke.append(found)
    longest = max(prepared_pairs, key=lambda item:len(item['gold_claim'])+len(item['candidate']))
    if longest not in selected_smoke:
        selected_smoke.append(longest)
    expected_smoke_ids = {item['pair_id'] for item in selected_smoke}
    for method in METHODS:
        score_path = require_file(out/(method+'_scores.json'))
        scores = load_json(score_path)
        if scores.get('complete') is not True or scores.get('method') != method or scores.get('input_sha256') != input_hash or set(scores['scores']) != expected_ids:
            raise ValueError('A complete provenance-bound method output is required: '+method)
        paths.add(score_path)
        if method in NEW_METHODS:
            for suffix in ['state.json', 'checkpoint.jsonl', 'resume_verification.json']:
                paths.add(require_file(out/(method+'_'+suffix)))
            proof = load_json(out/(method+'_resume_verification.json'))
            if proof.get('status') != 'passed' or proof.get('inference_calls') != 0 or proof.get('original_output_sha256') != digest(score_path):
                raise ValueError('Actual zero-inference resume proof is missing/stale: '+method)
            smoke_dir = out/'smoke'/method
            smoke_path = require_file(smoke_dir/(method+'_scores.json'))
            smoke = load_json(smoke_path)
            validation = smoke.get('smoke_validation') or {}
            if smoke.get('complete') is not True or smoke.get('smoke') is not True or validation.get('status') != 'passed':
                raise ValueError('Complete smoke parity evidence is required: '+method)
            if set(smoke['scores']) != expected_smoke_ids or not expected_smoke_ids:
                raise ValueError('Smoke pair coverage differs from the fixed selection: '+method)
            for field in ['worker_sha256', 'input_sha256', 'assets_sha256', 'environment_sha256']:
                if not smoke.get(field) or smoke[field] != scores.get(field):
                    raise ValueError('Smoke/formal provenance mismatch:'+method+':'+field)
            smoke_settings, formal_settings = dict(smoke['settings']), dict(scores['settings'])
            if smoke_settings.pop('smoke', None) is not True or formal_settings.pop('smoke', None) is not False or smoke_settings != formal_settings:
                raise ValueError('Smoke/formal settings differ beyond the smoke flag: '+method)
            tolerance = validation.get('absolute_tolerance')
            if type(tolerance) not in [int, float] or not 0 < tolerance <= 1e-5:
                raise ValueError('Smoke comparison tolerance must be at most 1e-5')
            comparisons = validation.get('comparisons', [])
            if len(comparisons) != len(expected_smoke_ids) or {item['pair_id'] for item in comparisons} != expected_smoke_ids:
                raise ValueError('Official-reference parity records do not cover all smoke pairs')
            for item in comparisons:
                pid = item['pair_id']
                error = abs(float(item['official_reference'])-float(smoke['scores'][pid]['score']))
                if not math.isfinite(error) or error > tolerance or abs(float(smoke['scores'][pid]['score'])-float(scores['scores'][pid]['score'])) > 1e-5:
                    raise ValueError('Smoke official/formal score parity failed: '+method)
            paths.add(smoke_path)
            for suffix in ['state.json', 'checkpoint.jsonl', 'resume_verification.json']:
                paths.add(require_file(smoke_dir/(method+'_'+suffix)))
            smoke_proof = load_json(smoke_dir/(method+'_resume_verification.json'))
            if smoke_proof.get('status') != 'passed' or smoke_proof.get('inference_calls') != 0 or smoke_proof.get('original_output_sha256') != digest(smoke_path):
                raise ValueError('Smoke zero-inference resume proof is missing/stale: '+method)
            if method == 'alignscore':
                parity_path = require_file(smoke_dir/'alignscore_optimization_parity.json')
                if digest(parity_path) != digest(out/'alignscore_optimization_validation.json') or digest(parity_path) != validation.get('optimization_parity',{}).get('sha256'):
                    raise ValueError('AlignScore optimization evidence differs from its scored smoke provenance')
                paths.add(parity_path)
            if method == 'moverscore_v2':
                for directory, payload in [(out, scores), (smoke_dir, smoke)]:
                    idf_path = require_file(directory/'moverscore_v2_idf.json')
                    if digest(idf_path) != payload.get('provenance', {}).get('idf', {}).get('sha256'):
                        raise ValueError('MoverScore IDF differs from scored provenance')
                    paths.add(idf_path)
    paths.update(supplemental_files(out))
    for path in (out/'figures').glob('*.png'):
        paths.add(path.resolve())
    if not any(path.parent == out/'figures' for path in paths):
        raise ValueError('Final report figures are required')

    # The v1 manifest is itself evidence; none of its original files is changed.
    import_proof = load_json(out/'legacy_import_verification.json')
    old_manifest_path = require_file(LEGACY/'delivery_manifest.json')
    if import_proof.get('status') != 'passed' or digest(old_manifest_path) != import_proof.get('source_manifest_sha256'):
        raise ValueError('Legacy source manifest does not match the verified import')
    old_manifest = load_json(old_manifest_path)
    paths.add(old_manifest_path)
    for name, item in old_manifest['contents'].items():
        path = require_file(PROJECT/name)
        if digest(path) != item['sha256']:
            raise ValueError('An immutable original evaluation artifact changed: '+name)
        paths.add(path)
    source_dir = Path(__file__).resolve().parent
    for path in source_dir.iterdir():
        if path.is_file() and (path.suffix in ['.py', '.md', '.txt', '.sh'] or path.name == 'THIRD_PARTY_NOTICES'):
            paths.add(path.resolve())
    paths.add(require_file(source_dir.parent/'run_patent_ce_extended_evaluation.py'))
    for path in paths:
        project_relative(path)
        if path.suffix.lower() in ['.zip', '.safetensors', '.bin', '.ckpt', '.pt', '.pth', '.pyc']:
            raise ValueError('Model/archive/cache files may not enter this result package: '+str(path))
    return sorted(paths, key=lambda path:project_relative(path).lower())


def package(output_dir):
    out = Path(output_dir).resolve()
    paths = select_files(out)
    manifest = dict(schema_version=2, created_at_utc=stamp(), status='verified_and_reviewed',
        contents={project_relative(path):dict(bytes=path.stat().st_size, sha256=digest(path)) for path in paths},
        source_dimensions=['Quality'], primary_methods=13, scored_settings=14,
        source_rows=1228, primary_predictions=15964, supplementary_predictions=1228,
        legacy_files='All entries of the immutable v1 delivery manifest plus that manifest are retained',
        exclusions=['Model weights', 'Virtual environments', '__pycache__', 'qa*/page PNGs',
                    'Asset download/error logs', 'Partial asset manifests', 'Previous ZIP archives'],
        readme=project_relative(Path(__file__).parent/'README.md'))
    manifest_path = out/'delivery_manifest.json'
    atomic_json(manifest_path, manifest)
    paths.append(manifest_path)
    target = out/ARCHIVE_NAME
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, project_relative(path))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(paths):
            raise ValueError('Archive integrity/entry count check failed')
    os.replace(temporary, target)
    (out/'archive_sha256.txt').write_text(digest(target)+'  '+target.name+'\n', encoding='utf-8')
    print(str(target), flush=True)
    print(f'{len(paths)} files; {target.stat().st_size} bytes; ZIP CRC passed', flush=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    package(parser.parse_args().output_dir)
