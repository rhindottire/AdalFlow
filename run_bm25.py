from adalflow.components.retriever.bm25_retriever import BM25Retriever

# 1) Siapkan dokumen
docs = [
    {"content": "Machine learning and data mining techniques"},
    {"content": "Natural language processing with BM25 ranking"},
    {"content": "AdalFlow makes building retrieval pipelines easy"},
    {"content": "Local Outlier Factor for anomaly detection in data"}
]

# 2) Inisialisasi BM25Retriever
bm25 = BM25Retriever(
    top_k=4,
    documents=docs,
    document_map_func=lambda d: d["content"],
)

print(bm25.tokenized_documents)
# 3) Lakukan pencarian
query   = "data"
outputs = bm25(input=query)    

# 4) Tampilkan hasil
print("Query:", query)
print("Query:", outputs)
# a) Cara 1: ambil elemen pertama
result = outputs[0]
for idx, score in zip(result.doc_indices, result.doc_scores):
    print(f"  - Doc #{idx}: {docs[idx]['content']} (score: {score:.4f})")

# b) Cara 2: iterasi semua (jika multiple queries)
# for result in outputs:
#     ...
