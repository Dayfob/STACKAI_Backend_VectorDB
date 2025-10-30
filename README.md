# Vector Database REST API

A high-performance REST API for vector similarity search with multiple indexing algorithms, built from scratch using FastAPI and Domain-Driven Design principles.

## Features

- **CRUD Operations** for Libraries, Documents, and Chunks
- **Multiple Vector Index Algorithms**:
  - Brute Force (exact search)
  - HNSW (Hierarchical Navigable Small World)
  - LSH (Locality-Sensitive Hashing)
- **k-NN Vector Search** with cosine similarity
- **Thread-Safe Operations** using Read-Write Locks
- **RESTful API** with OpenAPI/Swagger documentation
- **Docker Support** for easy deployment
- **Metadata Filtering** for advanced search
- **Persistence** options (in-memory and disk-based)

## Architecture

The project follows **Domain-Driven Design (DDD)** with a clean architecture:

```
src/
├── api/                    # API Layer (FastAPI)
│   ├── v1/routers/        # REST endpoints
│   └── main.py            # Application entry point
├── core/                   # Business Logic Layer
│   ├── services/          # Business logic services
│   ├── config.py          # Configuration
│   └── exceptions.py      # Custom exceptions
├── domain/                 # Domain Layer
│   ├── models/            # Domain entities
│   └── enums.py           # Domain enumerations
├── infrastructure/         # Infrastructure Layer
│   ├── indexes/           # Vector index implementations
│   ├── repositories/      # Data access layer
│   ├── persistence/       # Storage backends
│   └── concurrency/       # Thread-safety utilities
├── schemas/                # API Schemas (Request/Response)
└── utils/                  # Utilities (embeddings, math)
```

## Vector Index Algorithms

### 1. Brute Force Index

**Time Complexity:**
- Build: O(n)
- Search: O(n·d) where d = dimension
- Space: O(n·d)

**Pros:**
- Exact results (100% recall)
- Simple implementation
- No index building overhead

**Cons:**
- Slow for large datasets
- Not scalable

### 2. HNSW (Hierarchical Navigable Small World)

**Time Complexity:**
- Build: O(n·log(n)·d) amortized
- Search: O(log(n)·d) average
- Space: O(n·M·d) where M = max connections

**Pros:**
- Very fast search
- Good recall/accuracy trade-off
- Supports dynamic updates

**Cons:**
- Complex implementation
- Higher memory usage
- Approximate results

### 3. LSH (Locality-Sensitive Hashing)

**Time Complexity:**
- Build: O(n·L·k) where L = tables, k = hash functions
- Search: O(1) average, O(n) worst case
- Space: O(n·L)

**Pros:**
- Constant time search (average)
- Memory efficient
- Good for high-dimensional data

**Cons:**
- Approximate results
- Requires parameter tuning
- Quality depends on hash functions

## Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- Cohere API key ([Get one here](https://cohere.com))

### Docker Deployment

1. **Set up environment**
```bash
cp .env.example .env
# Edit .env and add your Cohere API key
```

2. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

3. **Access the API**
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

## Usage

### Create a Library

```bash
curl -X POST "http://localhost:8000/api/v1/libraries/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Library",
    "description": "A collection of documents",
    "index_type": "brute_force"
  }'
```

### Create a Document

```bash
curl -X POST "http://localhost:8000/api/v1/libraries/{library_id}/documents/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Paper",
    "metadata": {"author": "John Doe"}
  }'
```

### Create Chunks

```bash
curl -X POST "http://localhost:8000/api/v1/documents/{document_id}/chunks/" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning is a subset of AI.",
    "metadata": {"page": 1}
  }'
```

### Build Index

```bash
curl -X POST "http://localhost:8000/api/v1/libraries/{library_id}/index"
```

### Search

```bash
curl -X POST "http://localhost:8000/api/v1/libraries/{library_id}/search/" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What is machine learning?",
    "k": 5
  }'
```

## Testing

### Testing in Docker (Recommended)

```bash
# Run all tests with coverage
docker-compose -f docker-compose.test.yml up --build test

# Run only unit tests
docker-compose -f docker-compose.test.yml run --rm test-unit

# Run only integration tests
docker-compose -f docker-compose.test.yml run --rm test-integration

# Watch mode (auto-rerun on file changes)
docker-compose -f docker-compose.test.yml up test-watch

# Clean up test containers
docker-compose -f docker-compose.test.yml down
```

**View coverage report:**
```bash
# After running tests, open coverage report
# Windows
start htmlcov\index.html

# Linux
xdg-open htmlcov/index.html

# Mac
open htmlcov/index.html
```

### Test Coverage

- **100+ test cases** covering all functionality
- **Unit tests**: Indexes, utilities, storage, concurrency
- **Integration tests**: All API endpoints (libraries, documents, chunks, search)
- **E2E tests**: Complete workflows and error scenarios
- **Coverage target**: >90%

See [TESTING.md](TESTING.md) for detailed documentation.

## Technical Choices

### Why These Indexes?

1. **Brute Force**: Baseline for comparison, guarantees exact results
2. **HNSW**: Industry-standard for production vector search (used by Qdrant, Weaviate)
3. **LSH**: Educational value, demonstrates probabilistic data structures

### Concurrency Strategy

- **Read-Write Lock (RWLock)**: Allows multiple readers or single writer
- **Writer Priority**: Prevents writer starvation
- **Thread-Safe Repositories**: All data access is protected

### Persistence Approach

- **In-Memory**: Fast, good for development and testing
- **Disk Storage**: JSON/Pickle serialization for data persistence
- **Trade-offs**:
  - JSON: Slower, human-readable, portable
  - Pickle: Faster, binary, Python-specific

## Code Quality

### Linting
```bash
ruff check src/
```

### Formatting
```bash
black src/
```

### Type Checking
```bash
mypy src/
```

## Project Status

- [x] CRUD for Libraries, Documents, Chunks
- [x] Domain models and schemas
- [x] Infrastructure layer (repositories, persistence, concurrency)
- [x] FastAPI REST API
- [x] Docker support
- [x] Vector index implementations
- [x] Service layer business logic
- [x] Thread-safe operations
- [x] Embedding service integration
- [x] **Comprehensive test suite (100+ tests, unit + integration + E2E)**
- [x] **Docker-based testing with coverage reports**
- [ ] Metadata filtering (basic implementation done)
- [ ] Persistence to disk (basic implementation done)
- [ ] Benchmark script (template ready)

## Next Steps

1. ✅ Implement RWLock logic - **DONE**
2. ✅ Implement vector index algorithms (Brute Force, HNSW, LSH) - **DONE**
3. ✅ Implement math utilities - **DONE**
4. ✅ Implement embedding service - **DONE**
5. ✅ Implement storage logic - **DONE**
6. ✅ Implement repository logic - **DONE**
7. ✅ Implement service layer - **DONE**
8. ✅ Implement API endpoints - **DONE**
9. ✅ **Write comprehensive tests (100+ test cases)** - **DONE**
10. 🔄 Enhance metadata filtering capabilities
11. 🔄 Optimize disk persistence performance
12. 🔄 Create detailed benchmark comparisons
13. 🔄 Add batch operations for chunks
14. 🔄 Implement query result caching

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or feedback, please open an issue on GitHub.
