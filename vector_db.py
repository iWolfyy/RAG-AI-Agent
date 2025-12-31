import os
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, dims=768):
        # 1. Initialize Client - prioritize Env Vars for Cloud, fallback to Local
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        
        # 2. Use a consistent variable name (self.collection_name)
        self.collection_name = "docs" 
        self.dims = dims

        # 3. Auto-create collection if missing
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dims, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) 
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection_name, points=points)

    def search(self, query_vector, top_k: int = 5):
        # query_points is the modern, faster way to search in Qdrant
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector, 
            limit=top_k,
            with_payload=True
        ).points 

        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
    
    def delete_document(self, source_name: str):
        """Removes all chunks associated with a specific filename."""
        return self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source", 
                            match=models.MatchValue(value=source_name)
                        )
                    ]
                )
            )
        )
    
    def wipe_database(self):
        """Deletes the entire collection and recreates it empty."""
        # FIXED: Changed self.collection_name to match __init__
        self.client.delete_collection(collection_name=self.collection_name)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.dims, distance=Distance.COSINE)
        )