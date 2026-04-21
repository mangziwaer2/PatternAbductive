from utils.kg_hints import build_kg_hints_text
from utils.textualization import TEXT_CONDITION_FIELD, attach_textual_fields


def build_minimal_text_record(
        record,
        kg,
        sample_id,
        graph_split: str | None = None,
        include_kg_hints: bool = True,
        kg_hints_max_facts: int = 8):
    enriched = attach_textual_fields(record, kg)
    condition_text = enriched.get(TEXT_CONDITION_FIELD, '')
    observation_text = enriched['observation_text']
    kg_hints_text = ''

    if include_kg_hints and graph_split is not None and kg_hints_max_facts > 0:
        kg_hints_text = build_kg_hints_text(
            observation_text=observation_text,
            kg=kg,
            condition_text=condition_text,
            graph_split=graph_split,
            max_facts=kg_hints_max_facts,
        )

    return {
        'sample_id': int(sample_id),
        'pattern_str': record['pattern_str'],
        'condition_signature': record.get('condition_signature', 'unconditional') or 'unconditional',
        'observation_text': observation_text,
        'condition_text': condition_text,
        'kg_hints_text': kg_hints_text,
        'hypothesis_text': enriched['hypothesis_text'],
    }
