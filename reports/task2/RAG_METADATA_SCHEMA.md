# Foody RAG metadata schema

`data/foody/rag/documents.jsonl` contains one `id`, `text`, and `metadata` object per distinct valid `review_id`. `src/rag/vector_store.py::chroma_metadata` persists `review_id`, `restaurant_key`, `restaurant_name`, `review_timestamp`, `raw_rating`, `predicted_sentiment`, JSON-encoded `predicted_aspects`, JSON-encoded `detected_issues`, and `source=foody`. When a date exists it adds `timestamp_epoch`; each predicted aspect adds a Boolean `aspect_<name>` key.

Restaurant and date fields constrain retrieval; sentiment, aspect, and issue fields support filters and evidence checks. Ratings are source attributes, never gold sentiment labels. The restaurant key is derived from its name, not an official ID. Chunk simulations preserve all metadata but would require a separate chunk ID.
