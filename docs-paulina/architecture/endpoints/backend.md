## RAG backend

### POST /upload_documents

- **Description**: Upload new document
- **Flow**: 
1.Save file to object storage
2.Create document record
3.Chunk document
4.Generate embeddings
5.Index embeddings

### GET /list_documents
- **Description**: Return list of user documents

### DELETE /delete_document
- **Description**: Return list of user documents

### POST /q&a

Conversaciones persistentes?
### 