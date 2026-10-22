# Cleaning and labeling flow

```mermaid
flowchart LR
A[Raw source] --> B[Strict UTF-8 validation]
B --> C[Missing text handling]
C --> D[Unicode normalization]
D --> E[Duplicate grouping]
E --> F[Date/rating parsing where present]
F --> G[Schema and domain validation]
G --> H[Processed data]
H --> I[Foody model enrichment for demo/RAG]
```

VN-MSASar labels are supplied by the public dataset. Review-ID and normalized-text groups define the frozen split, without removing the 5,869 sentence/aspect rows. Foody is a first-party crawl; its AI cleaning verifies text/date/rating fields and creates deduplicated observations. Sentiment/aspect enrichment is model inference, not manual gold labeling. No human annotation or inter-annotator Kappa is claimed.
