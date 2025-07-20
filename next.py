import os
import tempfile
import shutil
import hashlib
import json
import time
import random
from typing import List, Dict, Set, Any
import networkx as nx
from pyvis.network import Network
from graphviz import Digraph
import openai
import traceback
import subprocess
import chromadb
from chromadb.utils import embedding_functions

# Configure OpenAI API
OPENAI_API_KEY = "your-openai-api-key"  # Replace with your actual key
openai.api_key = OPENAI_API_KEY

# Initialize ChromaDB with persistent client
chroma_client = chromadb.PersistentClient(path=".chromadb")
# Use OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_or_create_collection(
    name="architecture_docs",
    embedding_function=openai_ef
)

# Rate limiting and concurrency control for OpenAI
OPENAI_RATE_LIMIT = 2000  # tokens per minute
last_openai_call = 0
tokens_used = 0

def safe_openai_call(prompt: str, model: str = "gpt-4-turbo") -> str:
    """Safe OpenAI API call with exponential backoff and rate limiting"""
    global last_openai_call, tokens_used

    # Calculate tokens in prompt (approx)
    prompt_tokens = len(prompt) // 4
    current_time = time.time()
    
    # Rate limiting
    if current_time - last_openai_call < 60 and tokens_used + prompt_tokens > OPENAI_RATE_LIMIT:
        sleep_time = 60 - (current_time - last_openai_call)
        print(f"⚠️ OpenAI rate limit approaching. Sleeping for {sleep_time:.1f}s")
        time.sleep(sleep_time)
        tokens_used = 0

    retries = 7
    base_delay = 2  # base delay in seconds
    max_delay = 60  # maximum delay

    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000
            )
            last_openai_call = time.time()
            tokens_used += response.usage.total_tokens
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 1))
            print(f"⚠️ OpenAI rate limit exceeded (attempt {attempt+1}/{retries}), retrying in {delay:.1f}s")
            time.sleep(delay)
        except (openai.APIConnectionError, openai.APIError) as e:
            delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 1))
            print(f"⚠️ OpenAI connection issue (attempt {attempt+1}/{retries}): {str(e)}")
            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ OpenAI API error (attempt {attempt+1}/{retries}): {str(e)}")
            if attempt == retries - 1:
                print(f"⚠️ OpenAI API failed after {retries} retries")
                return ""
            delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, 1))
            time.sleep(delay)
    return ""

def parse_with_openai(file_path: str, content: str) -> Dict:
    """Parse file content using OpenAI API with enhanced prompt for Java 17+"""
    # Skip large non-Java files to save API calls
    if not file_path.endswith(('.java', '.xml', '.gradle', '.properties', '.yml', '.yaml')):
        return {"file_type": "SKIPPED", "path": file_path}

    # Truncate very large files while preserving structure
    truncated_content = content[:15000]  # Reduced for OpenAI context limits
    if len(content) > 15000:
        truncated_content += "\n\n... [CONTENT TRUNCATED] ..."

    prompt = f"""
    Analyze the following source file and extract comprehensive architectural information.
    The file path is: {file_path}

    IMPORTANT: This may be a modern Java 17+ file. Pay special attention to:
    - Records (record classes)
    - Sealed classes and interfaces
    - Pattern matching (instanceof, switch expressions)
    - Text blocks
    - New Java 17 language features
    - Modern framework annotations (Spring Boot 3+, Micronaut, Quarkus)

    File Content:
    {truncated_content}

    Return ONLY a JSON object with these fields:
    - file_type: "JAVA", "BUILD", "CONFIG", or "OTHER"
    - package: Package declaration (if Java)
    - imports: List of imports (if Java)
    - classes: List of classes/interfaces/records with:
      - name: Class name
      - type: "class", "interface", "enum", "annotation", "record"
      - annotations: List of annotations
      - methods: List of methods with:
          - name: Method name
          - parameters: List of parameter types
          - return_type: Return type
          - annotations: Method annotations
      - fields: List of fields with names and types
      - extends: Parent class
      - implements: List of interfaces
      - is_sealed: Boolean for sealed classes
      - permits: List of permitted subclasses (for sealed classes)
    - dependencies: List of dependencies (for build files)
    - configurations: Key-value pairs (for config files)
    - endpoints: List of REST endpoints with:
      - path: Endpoint path
      - method: HTTP method
      - controller: Owning class
    - kafka_listeners: List of Kafka listeners with:
      - topic: Topic name
      - group_id: Consumer group ID
    - framework_usage: List of detected frameworks (Spring, Kafka, etc.)
    - summary: Brief architectural summary of this file
    - java_features: List of Java language features used (records, sealed classes, pattern matching, etc.)

    For non-Java files, only include relevant fields. Do not include any explanations or additional text.
    """

    response = safe_openai_call(prompt)

    # Handle empty or invalid responses
    if not response:
        return {"error": "OpenAI API failed", "path": file_path}

    try:
        # Extract JSON from response if needed
        if response.startswith("```json"):
            response = response.split("```json")[1].split("```")[0].strip()
        return json.loads(response)
    except json.JSONDecodeError:
        print(f"⚠️ Failed to parse OpenAI response for {file_path}")
        print(f"Response content: {response[:500]}...")
        return {"error": "Invalid JSON response from OpenAI", "path": file_path}

def parse_file(file_path: str) -> Dict:
    """Parse any file type using OpenAI with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Skip binary files
        if '\0' in content:
            return {"file_type": "BINARY", "path": file_path}

        # Skip large files that are not source code
        if len(content) > 100000 and not file_path.endswith(('.java', '.kt', '.scala', '.groovy')):
            return {"file_type": "SKIPPED", "path": file_path, "reason": "Too large"}

        # Use OpenAI for parsing
        openai_result = parse_with_openai(file_path, content)

        # Add basic file info
        openai_result.update({
            "path": file_path,
            "size": len(content),
            "hash": hashlib.md5(content.encode()).hexdigest()
        })
        return openai_result
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {str(e)}")
        return {"error": str(e), "path": file_path}

def analyze_java_project(project_path: str) -> Dict:
    """Comprehensive project analysis using only OpenAI- sequential processing"""
    project_name = os.path.basename(project_path.rstrip(os.sep))

    # Collect all files
    all_files = []
for root, dirs, files in os.walk(project_path):
        # Skip common non-source directories
        skip_dirs = ["test", "target", "build", "node_modules", ".git", ".idea", "dist", "out", "bin"]
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        # Skip specific file types
        skip_exts = ['.md', '.txt', '.log', '.iml', '.gitignore', '.jar', '.class', '.png', '.jpg', '.gif']
        all_files.extend(
            os.path.join(root, f) for f in files
            if not f.startswith('.') and not any(f.endswith(ext) for ext in skip_exts)
        )

    print(f"🔍 Found {len(all_files)} files to analyze...")

    # Process files sequentially
    parsed_files = []
    for i, file_path in enumerate(all_files):

    # Detect project framework and architecture
    framework_info = detect_project_framework(parsed_files)

    # Build project structure
    project_structure = {
        "name": project_name,
        "path": project_path,
        "framework": framework_info.get("primary_framework", "Unknown"),
        "build_system": framework_info.get("build_system", "Unknown"),
        "architecture": framework_info.get("architecture", "Unknown"),
        "java_version": framework_info.get("java_version", "17+"),
        "technologies": framework_info.get("technologies", []),
        "files": parsed_files,
        "classes": [],
        "records": [],
        "sealed_classes": [],
        "controllers": [],
        "services": [],
        "repositories": [],
        "entities": [],
        "dependencies": set(),
        "kafka_listeners": [],
        "rest_endpoints": [],
        "framework_usage": {},
        "java_features": set()
    }

    # Aggregate data from all files
    for file_data in parsed_files:
        if "error" in file_data or file_data.get("file_type") == "SKIPPED":
            continue

        # Track Java features
        if "java_features" in file_data:
            project_structure["java_features"].update(file_data["java_features"])

        # Process Java files
        if file_data.get("file_type") == "JAVA":
            for cls in file_data.get("classes", []):
                # Classify by type
                if cls.get("type") == "record":
                    project_structure["records"].append(cls["name"])
                elif cls.get("is_sealed", False):
                    project_structure["sealed_classes"].append(cls["name"])
                else:
                    project_structure["classes"].append(cls["name"])

                # Classify by annotations
                for ann in cls.get("annotations", []):
                    if "Controller" in ann or "RestController" in ann:
                        project_structure["controllers"].append(cls["name"])
                    elif "Service" in ann:
                        project_structure["services"].append(cls["name"])
                    elif "Repository" in ann:
                        project_structure["repositories"].append(cls["name"])
                    elif "Entity" in ann:
                        project_structure["entities"].append(cls["name"])

            # Collect dependencies
            project_structure["dependencies"].update(file_data.get("imports", []))

        # Process build files
        elif file_data.get("file_type") == "BUILD":
            project_structure["dependencies"].update(file_data.get("dependencies", []))

        # Collect framework usage
        for framework in file_data.get("framework_usage", []):
            if framework not in project_structure["framework_usage"]:
                project_structure["framework_usage"][framework] = []
            project_structure["framework_usage"][framework].append(file_data["path"])

        # Collect endpoints and listeners
        project_structure["rest_endpoints"].extend(file_data.get("endpoints", []))
        project_structure["kafka_listeners"].extend(file_data.get("kafka_listeners", []))

    # Convert sets to lists
    project_structure["dependencies"] = list(project_structure["dependencies"])
    project_structure["java_features"] = list(project_structure["java_features"])

    # Generate comprehensive summary
    project_structure["summary"] = generate_project_summary(project_structure)

    return project_structure

def detect_project_framework(files: List[Dict]) -> Dict:
    """Use OpenAI to analyze the entire project's framework"""
    context = "\n".join(
        f"File: {f['path']}\nSummary: {f.get('summary', 'No summary')}\nJava Features: {', '.join(f.get('java_features', []))}"
        for f in files if f.get('summary')
    )[:10000]  # Reduced context size

    prompt = f"""
    Analyze this Java project's structure and determine:
    - Primary framework (Spring Boot 3+, Quarkus, Micronaut, etc.)
    - Estimated framework version
    - Build system (Maven, Gradle)
    - Database technologies
    - Messaging systems (Kafka, JMS, etc.)
    - Architectural pattern (Layered, Microservices, etc.)
    - Java version used (based on language features)

    Project Files Context:
    {context}

    Return ONLY a JSON object with:
    - primary_framework: Name and version
    - build_system: "Maven" or "Gradle"
    - technologies: List of key technologies
    - architecture: Architectural pattern
    - java_version: Estimated Java version (e.g., "17", "21")
    - confidence: Confidence score (0-100)
    """

    response = safe_openai_call(prompt)
    if not response:
        return {"primary_framework": "Unknown", "java_version": "17+"}

    try:
        if response.startswith("```json"):
            response = response.split("```json")[1].split("```")[0].strip()
        return json.loads(response)
    except:
        return {"primary_framework": "Unknown", "java_version": "17+"}

def generate_project_summary(project: Dict) -> str:
    """Generate detailed project summary using OpenAI"""
    context = f"""
    Project Name: {project['name']}
    Java Version: {project['java_version']}
    Framework: {project['framework']}
    Architecture: {project['architecture']}
    Build System: {project['build_system']}
    Technologies: {', '.join(project['technologies'])}
    Java Features: {', '.join(project['java_features'])}
    Files: {len(project['files'])}
    Classes: {len(project['classes'])}
    Records: {len(project['records'])}
    Sealed Classes: {len(project['sealed_classes'])}
    Controllers: {len(project['controllers'])}
    Services: {len(project['services'])}
    Repositories: {len(project['repositories'])}
    Entities: {len(project['entities'])}
    REST Endpoints: {len(project['rest_endpoints'])}
    Kafka Listeners: {len(project['kafka_listeners'])}
    """

    prompt = f"""
    As a software architect, generate a comprehensive technical report based on this modern Java project architecture:

    {context}

    Structure your report with these sections:
    1. System Overview
    2. Architectural Patterns (Microservices, Modular, etc.)
    3. Java Version Analysis (features used, compatibility)
    4. Technology Stack Analysis (frameworks, databases, messaging)
    5. Component Interactions
    6. Modern Java Features Usage (records, sealed classes, pattern matching)
    7. REST API Design Analysis
    8. Asynchronous Processing (Kafka, etc.)
    9. Deployment Architecture
    10. Scaling Characteristics
    11. Security Considerations
    12. Potential Improvements

    Provide deep technical insights and professional analysis.
    """

    return safe_openai_call(prompt) or "Summary generation failed"

def build_knowledge_graph(project: Dict) -> nx.DiGraph:
    """Build a comprehensive knowledge graph using only Gemini-derived data"""
    G = nx.DiGraph()

    # Add project node
    G.add_node(project['name'],
               type='project',
               framework=project['framework'],
               java_version=project['java_version'],
               architecture=project['architecture'])

    # Add all files
    for file_data in project['files']:
        if "error" in file_data or file_data.get("file_type") == "SKIPPED":
            continue

        file_id = f"{project['name']}/{file_data['path']}"
        G.add_node(file_id,
                   type='file',
                   file_type=file_data.get('file_type', 'UNKNOWN'),
                   summary=file_data.get('summary', ''),
                   java_features=', '.join(file_data.get('java_features', [])))
        G.add_edge(project['name'], file_id, relationship='contains')

        # Link frameworks
        for framework in file_data.get('framework_usage', []):
            G.add_node(framework, type='framework')
            G.add_edge(file_id, framework, relationship='uses')

        # Process Java classes
        if file_data.get('file_type') == 'JAVA':
            for cls in file_data.get('classes', []):
                class_id = f"{project['name']}/{cls['name']}"
                node_type = 'record' if cls.get('type') == 'record' else 'class'
                G.add_node(class_id,
                           type=node_type,
                           annotations=', '.join(cls.get('annotations', [])),
                           is_sealed=cls.get('is_sealed', False))
                G.add_edge(file_id, class_id, relationship='defines')

                # Inheritance
                if cls.get('extends'):
                    parent_id = f"{project['name']}/{cls['extends']}"
                    G.add_node(parent_id, type='class')
                    G.add_edge(class_id, parent_id, relationship='extends')

                # Implementation
                for interface in cls.get('implements', []):
                    interface_id = f"{project['name']}/{interface}"
                    G.add_node(interface_id, type='interface')
                    G.add_edge(class_id, interface_id, relationship='implements')

                # Permitted subclasses for sealed classes
                if cls.get('is_sealed', False):
                    for permitted in cls.get('permits', []):
                        permitted_id = f"{project['name']}/{permitted}"
                        G.add_node(permitted_id, type='class')
                        G.add_edge(class_id, permitted_id, relationship='permits')

        # Link dependencies
        for dep in file_data.get('imports', []) + file_data.get('dependencies', []):
            G.add_node(dep, type='dependency')
            G.add_edge(file_id, dep, relationship='depends_on')

    # Add REST endpoints
    for endpoint in project.get('rest_endpoints', []):
        endpoint_id = f"{project['name']}/endpoint/{endpoint.get('path', '')}"
    G.add_node(endpoint_id,
                   type='endpoint',
                   method=endpoint.get('method', 'GET'))
        G.add_edge(project['name'], endpoint_id, relationship='exposes')

        if endpoint.get('controller'):
            controller_id = f"{project['name']}/{endpoint['controller']}"
            if controller_id in G:
                G.add_edge(controller_id, endpoint_id, relationship='handles')

    # Add Kafka listeners
    for listener in project.get('kafka_listeners', []):
        listener_id = f"{project['name']}/kafka/{listener.get('topic', '')}"
    G.add_node(listener_id,
                   type='kafka_listener',
                   topic=listener.get('topic', ''),
                   group_id=listener.get('group_id', ''))
        G.add_edge(project['name'], listener_id, relationship='consumes')

    return G

def visualize_knowledge_graph(graph: nx.DiGraph, output_file: str = "knowledge_graph.html"):
    """Enhanced interactive visualization with Java 17+ features"""
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")

    # Color mapping        
    colors = {
        'project': '#FF6B6B',
        'file': '#4ECDC4',
        'class': '#FFD166',
        'record': '#FF9E6D',  # Distinct color for records
        'interface': '#06D6A0',
        'dependency': '#118AB2',
        'framework': '#073B4C',
        'endpoint': '#EF476F',
        'kafka_listener': '#9B5DE5',
        'sealed': '#C77DFF'  # Color for sealed classes
    }

    # Add nodes
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'class')
        color = colors.get(node_type, '#888888')

        # Special styling for sealed classes and records
        shape = 'dot'
        if node_type == 'record':
            shape = 'diamond'
        elif data.get('is_sealed', False):
            shape = 'star'
            color = colors.get('sealed', '#C77DFF')

        title = f"<b>{node}</b><br>Type: {node_type}"
        for k, v in data.items():
            if k != 'type' and v:
                title += f"<br>{k}: {v}"

        net.add_node(node, label=node, color=color, title=title, size=25, shape=shape)

    # Add edges
    for src, dst, data in graph.edges(data=True):
        rel = data.get('relationship', 'related')
        color = '#AAAAAA'
        dashes = False

        # Special edge styling
        if rel == 'permits':
            color = '#C77DFF'
            dashes = True
        elif rel == 'depends_on':
            color = '#FF6B6B'
        elif rel == 'extends':
            color = '#FFD166'

        net.add_edge(src, dst, title=rel, color=color, dashes=dashes)

    # Customize physics
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "springLength": 150
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    net.show(output_file)
def index_project_documents(project: Dict):
    """Index all project documents in ChromaDB"""
    documents = []
    metadatas = []
    ids = []

    # Index project summary
    if project.get('summary'):
        documents.append(project['summary'])
        metadatas.append({"type": "project_summary", "project": project['name']})
        ids.append(f"summary_{project['name']}")

    # Index individual files
    for file_data in project['files']:
        if "error" in file_data or file_data.get("file_type") == "SKIPPED":
            continue

        content = f"{file_data.get('summary', '')}\n{json.dumps(file_data, indent=2)}"
        documents.append(content)
        metadatas.append({
            "type": "file",
            "project": project['name'],
            "path": file_data['path'],
            "file_type": file_data.get('file_type', 'UNKNOWN'),
            "java_features": ', '.join(file_data.get('java_features', []))
        })
        ids.append(f"file_{file_data['path'].replace('/', '_')}")

    # Add to ChromaDB
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
class ProjectAnalyzer:
    def __init__(self, project: Dict):
        self.project = project
        self.knowledge_graph = build_knowledge_graph(project)
        index_project_documents(project)

    def query(self, question: str, top_k: int = 5) -> Dict:
        """Query the project knowledge base"""
        # First search ChromaDB
        try:
            results = collection.query(
                query_texts=[question],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"⚠️ ChromaDB query failed: {str(e)}")
            results = {'documents': [[]], 'metadatas': [[]]}

        # Prepare context
        context = "Relevant project information:\n"
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                context += f"\nFrom {meta.get('type', 'unknown')}"
                if 'path' in meta:
                    context += f" ({meta['path']})"
                context += f":\n{doc}\n"
        else:
            context += "No relevant documents found in ChromaDB\n"

        # Enhance with knowledge graph info
        graph_context = "\nKey architectural relationships:\n"
        try:
            for node in nx.topological_sort(self.knowledge_graph):
                if question.lower() in node.lower():
                    neighbors = list(self.knowledge_graph.neighbors(node))
                    if neighbors:
                        graph_context += f"- {node} relates to: {', '.join(neighbors)}\n"
        except Exception as e:
            graph_context += f"Knowledge graph error: {str(e)}\n"

        # Generate comprehensive answer using OpenAI
        prompt = f"""
        As an expert software architect, answer this question about the project:
        Question: {question}

        Context from project analysis:
        {context}

        {graph_context}

        Provide a detailed response with:
        1. Direct answer
        2. Architectural implications
        3. Relevant components
        4. Technology considerations
        5. Modern Java feature usage
        6. Potential improvements

        Format with clear section headings.
        """

        answer = safe_openai_call(prompt) or "Could not generate answer"

        return {
            "answer": answer,
            "sources": results.get('metadatas', [[]])[0] if results else []
        }
def clone_repository(repo_url: str) -> str:
    """Clone a git repository with error handling and subdirectory support"""
    try:
        # Extract repository name from URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')

        # Create temporary directory for cloning
        clone_dir = tempfile.mkdtemp(prefix=f"{repo_name}_")

        # Correct URL format if needed
        if "github.com" in repo_url and "/tree/" in repo_url:
            # Convert GitHub tree URL to raw repository URL
            repo_url = repo_url.split('/tree/')[0].replace('github.com', 'github.com') + ".git"

        print(f"Cloning {repo_url}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, clone_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return clone_dir
    except Exception as e:
        print(f"Failed to clone repository: {str(e)}")
        raise

def analyze_repository(repo_url: str) -> ProjectAnalyzer:
    """Full analysis pipeline for a git repository with subdirectory support"""
    try:
        print("🚀 Starting analysis...")

        # Clone repository
        repo_path = clone_repository(repo_url)
        print(f"✅ Repository cloned to: {repo_path}")
# Check if we need to analyze a subdirectory
        if "tree" in repo_url:
            # Extract subdirectory path from original URL
            subdir = repo_url.split("tree/")[1].split("/", 1)[1]
            project_path = os.path.join(repo_path, subdir)
            print(f"🔍 Using subdirectory: {subdir}")
        else:
            project_path = repo_path

        # Analyze project
        print("🔍 Analyzing project structure...")
        project = analyze_java_project(project_path)

        # Generate outputs if we have data
        if project.get('files'):
            print("📊 Generating visualizations...")
            try:
                visualize_knowledge_graph(
                    build_knowledge_graph(project),
                    f"{project['name']}_knowledge_graph.html"
                )
            except Exception as e:
                print(f"⚠️ Failed to generate knowledge graph: {str(e)}")

            # Generate architecture diagram
            try:
                dot = Digraph(comment=project['name'], format='png')
                dot.attr(rankdir='TB', labelloc='t', label=f"{project['name']} Architecture (Java {project['java_version']})")
            # Add layers with modern Java components
                with dot.subgraph(name='cluster_web') as web:
                    web.attr(label='Web Layer', style='filled', color='lightblue')
                    for controller in project['controllers'][:3]:
                        web.node(controller, shape='box3d')

                with dot.subgraph(name='cluster_service') as service:
                    service.attr(label='Service Layer', style='filled', color='lightgreen')
                    for svc in project['services'][:3]:
                        service.node(svc, shape='box')
                    for record in project['records'][:2]:
                        service.node(record, shape='component', color='orange')

                with dot.subgraph(name='cluster_data') as data:
                    data.attr(label='Data Layer', style='filled', color='lightyellow')
                    for repo in project['repositories'][:2]:
                        data.node(repo, shape='cylinder')
                    for entity in project['entities'][:3]:
                        data.node(entity, shape='ellipse')
                    for sealed in project['sealed_classes'][:1]:
                        data.node(sealed, shape='doublecircle', color='purple')

                # Framework specific components
                if 'Spring' in project.get('framework_usage', {}):
                    dot.node('Spring Boot', shape='component', color='lightpink')

                if 'Kafka' in project.get('framework_usage', {}):
                    dot.node('Kafka', shape='rect', color='lavender')

                # Add relationships
                if project['controllers'] and project['services']:
                    dot.edge(project['controllers'][0], project['services'][0], label='calls')

                if project['services'] and project['repositories']:
                    dot.edge(project['services'][0], project['repositories'][0], label='uses')

                if 'Kafka' in project.get('framework_usage', {}) and project['services']:
                    dot.edge(project['services'][0], 'Kafka', label='publishes', style='dashed')

                diagram_file = f"{project['name']}_architecture"
                dot.render(diagram_file, cleanup=True)
                print(f"✅ Generated diagram: {diagram_file}.png")
            except Exception as e:
                print(f"⚠️ Failed to generate architecture diagram: {str(e)}")
        else:
            print("⚠️ No files processed, skipping visualizations")

        print("✅ Analysis complete!")
        return ProjectAnalyzer(project)

    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    # Example modern Java repository
    REPO_URL = "https://github.com/spring-guides/gs-rest-service"

    analyzer = analyze_repository(REPO_URL)

    if analyzer:
        # Example queries
        questions = [
            "What modern Java features are used in this project?",
            "How are records utilized in the architecture?",
            "Explain the use of sealed classes in this system",
            "What is the REST API design pattern?",
            "How does the project handle asynchronous processing?",
            "What improvements would you suggest for better utilization of Java 17 features?"
        ]

        for q in questions:
            print(f"\nQuestion: {q}")
            result = analyzer.query(q)
            print(f"Answer:\n{result['answer']}\n")
            if result['sources']:
                print("Sources:")
                for src in result['sources']:
                    print(f"- {src.get('type', 'unknown')}: {src.get('path', '')}")
            else:
                print("No sources found")
    else:
        print("Analysis failed")