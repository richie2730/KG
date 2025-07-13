import os
import tempfile
from dotenv import load_dotenv
load_dotenv()
import shutil
import hashlib
import re
import ast
import json
from typing import List, Dict, Tuple, Set
from pathlib import Path
import javalang
from javalang.tree import ClassDeclaration, MethodDeclaration, FieldDeclaration, Annotation, PackageDeclaration
import networkx as nx
from pyvis.network import Network
from graphviz import Digraph
import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')


# Known Spring annotations for better recognition
SPRING_ANNOTATIONS = {
    "Controller", "RestController", "Service", "Repository", "Component",
    "Autowired", "Qualifier", "Value", "Bean", "Configuration",
    "EnableAutoConfiguration", "SpringBootApplication", "RequestMapping",
    "GetMapping", "PostMapping", "PutMapping", "DeleteMapping", "Entity"
}

# Known Kafka annotations
KAFKA_ANNOTATIONS = {"KafkaListener", "Topic", "KafkaHandler"}

def parse_java_file(file_path: str) -> Dict:
    """Enhanced Java parser that extracts detailed class information"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    try:
        tree = javalang.parse.parse(content)
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return {
            "package": "",
            "classes": [],
            "imports": [],
            "methods": [],
            "annotations": [],
            "fields": [],
            "rest_endpoints": []
        }

    package = tree.package.name if tree.package else ""
    imports = [imp.path for imp in tree.imports] if tree.imports else []
    classes = []
    methods = []
    annotations = []
    fields = []
    rest_endpoints = []

    # Extract detailed class information
    for path, node in tree:
        if isinstance(node, ClassDeclaration):
            class_info = {
                "name": node.name,
                "type": "class",
                "modifiers": node.modifiers or [],
                "annotations": [ann.name for ann in (node.annotations or [])],
                "methods": [],
                "fields": [],
                "extends": node.extends.name if node.extends else None,
                "implements": [impl.name for impl in node.implements] if node.implements else []
            }

            # Extract methods and fields
            for body_item in node.body:
                if isinstance(body_item, MethodDeclaration):
                    method_info = {
                        "name": body_item.name,
                        "return_type": body_item.return_type.name if body_item.return_type else "void",
                        "modifiers": body_item.modifiers or [],
                        "annotations": [ann.name for ann in (body_item.annotations or [])],
                        "parameters": [
                            {"type": param.type.name, "name": param.name}
                            for param in (body_item.parameters or [])
                        ]
                    }
                    class_info["methods"].append(method_info)
                    methods.append(body_item.name)

                    # Extract REST endpoints
                    for ann in method_info["annotations"]:
                        if "Mapping" in ann:
                            endpoint = {
                                "class": node.name,
                                "method": method_info["name"],
                                "annotation": ann,
                                "url": ann.replace("Mapping", "").lower() + "_path"
                            }
                            rest_endpoints.append(endpoint)

                elif isinstance(body_item, FieldDeclaration):
                    for declarator in body_item.declarators:
                        field_info = {
                            "name": declarator.name,
                            "type": body_item.type.name,
                            "modifiers": body_item.modifiers or [],
                            "annotations": [ann.name for ann in (body_item.annotations or [])]
                        }
                        class_info["fields"].append(field_info)
                        fields.append(declarator.name)

            classes.append(class_info)

        elif isinstance(node, Annotation):
            annotations.append(node.name)

    return {
        "package": package,
        "classes": classes,
        "imports": imports,
        "methods": methods,
        "annotations": annotations,
        "fields": fields,
        "rest_endpoints": rest_endpoints
    }

def clone_github_repo(repo_url: str) -> str:
    """Clone GitHub repository with better error handling"""
    try:
        if "github.com" not in repo_url:
            raise ValueError("Invalid GitHub URL")

        parts = repo_url.split("github.com/")[1].split("/")
        owner = parts[0]
        repo_name = parts[1].replace(".git", "")

        temp_dir = tempfile.mkdtemp()
        print(f"Cloning repository: {repo_url} to {temp_dir}")
        os.system(f"git clone --depth 1 {repo_url} {temp_dir}")

        return temp_dir
    except Exception as e:
        print(f"Error cloning repository: {str(e)}")
        raise

def is_spring_project(project_path: str) -> bool:
    """Check if project is a Spring Boot project"""
    # Check for build files
    build_files = ["pom.xml", "build.gradle", "build.gradle.kts"]
    for file in build_files:
        if os.path.exists(os.path.join(project_path, file)):
            return True

    # Check for Spring application class
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".java"):
                with open(os.path.join(root, file), 'r', errors='ignore') as f:
                    content = f.read()
                    if "@SpringBootApplication" in content:
                        return True
    return False

def analyze_java_project(project_path: str) -> Dict:
    """Robust project analysis with Spring detection and business domain recognition"""
    project_name = os.path.basename(project_path)
    is_spring = is_spring_project(project_path)
    business_domain = ""

    project_structure = {
        "name": project_name,
        "path": project_path,
        "is_spring": is_spring,
        "business_domain": "",
        "files": [],
        "classes": [],
        "controllers": [],
        "services": [],
        "repositories": [],
        "entities": [],
        "dependencies": set(),
        "kafka_topics": set(),
        "rest_endpoints": [],
        "class_relationships": {"inheritance": [], "implementation": []}
    }

    # Domain recognition patterns
    domain_keywords = {
        "pet": "Pet Care/Veterinary",
        "customer": "CRM/Retail",
        "order": "E-commerce",
        "inventory": "Supply Chain",
        "financial": "Banking/Finance",
        "medical": "Healthcare",
        "clinic": "Healthcare",
        "owner": "CRM"
    }

    # Walk through the project directory
    for root, dirs, files in os.walk(project_path):
        # Skip common non-source directories
        if any(x in root for x in ["test", "target", "build", "node_modules", ".git"]):
            continue

        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_path)

                try:
                    parsed_data = parse_java_file(file_path)

                    # Business domain detection
                    content = ""
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()

                    for keyword, domain in domain_keywords.items():
                        if re.search(rf'\b{keyword}\b', content, re.IGNORECASE):
                            project_structure["business_domain"] = domain
                            break

                    # Classify components
                    for class_info in parsed_data["classes"]:
                        class_name = class_info["name"]
                        project_structure["classes"].append(class_name)

                        # Spring component detection
                        if is_spring:
                            for ann in class_info["annotations"]:
                                if ann in SPRING_ANNOTATIONS:
                                    if "Controller" in ann or "RestController" in ann:
                                        project_structure["controllers"].append(class_name)
                                    elif "Service" in ann:
                                        project_structure["services"].append(class_name)
                                    elif "Repository" in ann:
                                        project_structure["repositories"].append(class_name)
                                    elif "Entity" in ann:
                                        project_structure["entities"].append(class_name)

                        # Kafka detection
                        for ann in class_info["annotations"]:
                            if ann in KAFKA_ANNOTATIONS:
                                project_structure["kafka_topics"].add(class_name + "_Topic")

                        # Class relationships
                        if class_info["extends"]:
                            project_structure["class_relationships"]["inheritance"].append(
                                (class_name, class_info["extends"])
                            )

                        for interface in class_info["implements"]:
                            project_structure["class_relationships"]["implementation"].append(
                                (class_name, interface)
                            )

                    # REST endpoints
                    project_structure["rest_endpoints"].extend(parsed_data["rest_endpoints"])

                    # Update project structure
                    file_info = {
                        "path": relative_path,
                        "package": parsed_data["package"],
                        "classes": [c["name"] for c in parsed_data["classes"]],
                        "imports": parsed_data["imports"],
                        "methods": parsed_data["methods"],
                        "annotations": parsed_data["annotations"],
                        "fields": parsed_data["fields"]
                    }

                    project_structure["files"].append(file_info)
                    project_structure["dependencies"].update(parsed_data["imports"])

                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    return project_structure

def generate_project_summary(project: Dict) -> str:
    """Generate detailed project summary using Gemini"""
    prompt = f"""
    Analyze this Java project structure and generate a comprehensive summary:

    Project Name: {project['name']}
    Business Domain: {project['business_domain']}
    Spring Project: {'Yes' if project['is_spring'] else 'No'}

    Key Components:
    - Controllers: {project['controllers']}
    - Services: {project['services']}
    - Repositories: {project['repositories']}
    - Entities: {project['entities']}

    REST Endpoints: {len(project['rest_endpoints'])}
    Kafka Topics: {len(project['kafka_topics'])}
    Dependencies: {len(project['dependencies'])}

    Please provide:
    1. Business domain analysis based on the components
    2. Architectural pattern recognition (MVC, Microservices, etc.)
    3. Technology stack identification
    4. Input, Output & Processing details
    5. Kafka Topics involved
    6. Data flow between key components
    7. Deployment characteristics
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return f"Summary generation failed: {str(e)}"

def build_knowledge_graph(projects: List[Dict]) -> nx.DiGraph:
    """Build a comprehensive knowledge graph with relationships"""
    G = nx.DiGraph()

    for project in projects:
        # Add project node with attributes
        G.add_node(project['name'],
                  type='project',
                  size=len(project['files']),
                  spring=project['is_spring'],
                  domain=project['business_domain'])

        # Add classes with types
        for cls in project['classes']:
            node_type = 'class'
            if cls in project['controllers']:
                node_type = 'controller'
            elif cls in project['services']:
                node_type = 'service'
            elif cls in project['repositories']:
                node_type = 'repository'
            elif cls in project['entities']:
                node_type = 'entity'

            G.add_node(cls, type=node_type)
            G.add_edge(project['name'], cls, relationship='contains')

        # Add dependencies
        for dep in project['dependencies']:
            G.add_node(dep, type='dependency')
            G.add_edge(project['name'], dep, relationship='depends_on')

        # Add REST endpoints
        for endpoint in project['rest_endpoints']:
            endpoint_id = f"{endpoint['class']}.{endpoint['method']}"
            G.add_node(endpoint_id, type='endpoint', method=endpoint['annotation'])
            G.add_edge(endpoint['class'], endpoint_id, relationship='exposes')

        # Add Kafka topics
        for topic in project['kafka_topics']:
            G.add_node(topic, type='kafka_topic')
            G.add_edge(project['name'], topic, relationship='uses_messaging')

        # Add class relationships
        for rel_type, relationships in project['class_relationships'].items():
            for relationship in relationships:
                if isinstance(relationship, (list, tuple)) and len(relationship) == 2:
                    source, target = relationship
                    if source in project['classes'] and target:
                        G.add_edge(source, target, relationship=rel_type)

    # Connect projects through shared dependencies
    for i, proj1 in enumerate(projects):
        for j, proj2 in enumerate(projects[i+1:], i+1):
            common_deps = proj1['dependencies'] & proj2['dependencies']
            for dep in common_deps:
                G.add_edge(proj1['name'], proj2['name'],
                          relationship='shares_dependency',
                          dependency=dep)

    return G

def visualize_knowledge_graph(graph: nx.DiGraph):
    """Enhanced visualization with component types"""
    net = Network(notebook=True, height="800px", width="100%", bgcolor="#222222", font_color="white")

    # Color mapping for node types
    type_colors = {
        'project': '#FF6B6B',
        'controller': '#4ECDC4',
        'service': '#FFD166',
        'repository': '#06D6A0',
        'entity': '#118AB2',
        'class': '#073B4C',
        'dependency': '#EF476F',
        'endpoint': '#9B5DE5',
        'kafka_topic': '#F15BB5'
    }

    # Add nodes with styling
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'class')
        color = type_colors.get(node_type, '#888888')
        title = f"<b>{node}</b><br>Type: {node_type}"

        if node_type == 'project':
            title += f"<br>Domain: {data.get('domain', '')}"
            size = 35
        elif node_type in ('controller', 'service'):
            size = 25
        else:
            size = 15

        net.add_node(node, label=node, color=color, title=title, size=size)

    # Add edges with styling
    for source, target, data in graph.edges(data=True):
        rel = data.get('relationship', 'related')
        color = '#FFFFFF'
        dashes = False

        if rel == 'contains':
            color = '#AAAAAA'
        elif rel == 'depends_on':
            color = '#FF6B6B'
            dashes = True
        elif rel == 'exposes':
            color = '#4ECDC4'
        elif rel == 'uses_messaging':
            color = '#F15BB5'

        net.add_edge(source, target, title=rel, color=color, dashes=dashes)

    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {
          "size": 16,
          "face": "arial"
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "smooth": false
      },
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

    net.show("knowledge_graph.html", notebook=False)

class ArchitectureRAG:
    def __init__(self, projects: List[Dict], knowledge_graph: nx.DiGraph):
        self.projects = projects
        self.knowledge_graph = knowledge_graph
        self.vector_index = {}
        self._build_vector_index()

    def _build_vector_index(self):
        """Reuse existing summaries instead of generating new ones"""
        for project in self.projects:
            # Use pre-generated summary if available
            if 'summary' in project:
                doc = project['summary']
            else:
                doc = generate_project_summary(project)
                
# class ArchitectureRAG:
#     def __init__(self, projects: List[Dict], knowledge_graph: nx.DiGraph):
#         self.projects = projects
#         self.knowledge_graph = knowledge_graph
#         self.vector_index = {}
#         self._build_vector_index()

#     def _build_vector_index(self):
#         """Build a vector index with detailed project information"""
#         for project in self.projects:
#             doc = generate_project_summary(project)
#             doc_hash = hashlib.md5(doc.encode()).hexdigest()
#             self.vector_index[doc_hash] = {
#                 "content": doc,
#                 "metadata": {
#                     "type": "project_summary",
#                     "project": project['name'],
#                     "domain": project['business_domain'],
#                     "spring": project['is_spring'],
#                     "components": {
#                         "controllers": project['controllers'],
#                         "services": project['services'],
#                         "repositories": project['repositories'],
#                         "entities": project['entities']
#                     }
#                 }
#             }

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Enhanced query with context-aware response"""
        # Step 1: Use Gemini to understand the query
        prompt = f"""
        Analyze this architecture question and extract key elements:

        Question: {question}

        Identify:
        1. Project components mentioned (controllers, services, etc.)
        2. Architectural concerns (scaling, security, data flow)
        3. Technology references (Spring, Kafka, databases)
        4. Business domain aspects

        Return your analysis as a well-formatted JSON object with keys: components, concerns, technologies, domain
        """

        try:
            response = model.generate_content(prompt)
            # Handle cases where response might not be valid JSON
            try:
                query_analysis = json.loads(response.text)
            except json.JSONDecodeError:
                # If response isn't JSON, create a default analysis
                query_analysis = {
                    "components": [],
                    "concerns": [],
                    "technologies": [],
                    "domain": ""
                }
        except Exception as e:
            print(f"Error analyzing query: {str(e)}")
            query_analysis = {
                "components": [],
                "concerns": [],
                "technologies": [],
                "domain": ""
            }

        # Step 2: Find relevant context
        relevant_docs = []
        for doc_data in self.vector_index.values():
            doc_text = doc_data["content"].lower()
            match_score = 0

            # Match components
            for comp in query_analysis.get("components", []):
                if comp.lower() in doc_text:
                    match_score += 1

            # Match technologies
            for tech in query_analysis.get("technologies", []):
                if tech.lower() in doc_text:
                    match_score += 1

            # Match domain
            if query_analysis.get("domain", "").lower() in doc_text:
                match_score += 2

            if match_score > 0:
                relevant_docs.append((match_score, doc_data))

        # Sort by relevance
        relevant_docs.sort(key=lambda x: x[0], reverse=True)
        relevant_docs = [doc[1] for doc in relevant_docs[:top_k]]

        # Step 3: Generate comprehensive answer
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        prompt = f"""
        Based on the following architectural context, answer the question: {question}

        Context:
        {context}

        Please provide:
        1. Direct answer to the question
        2. References to specific components
        3. Architectural implications
        4. Technology-specific considerations
        5. Any identified limitations

        Format your response as a clear, well-structured text with appropriate section headings.
        """

        try:
            answer = model.generate_content(prompt)
            return {
                "answer": answer.text,
                "context_sources": [doc["metadata"]["project"] for doc in relevant_docs]
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context_sources": []
            }

def generate_architecture_diagram(project: Dict) -> Digraph:
    """Generate detailed architecture diagram"""
    dot = Digraph(comment=project['name'], format='png')
    dot.attr(rankdir='TB', labelloc='t', label=f"{project['name']} Architecture")

    # Cluster for layers
    with dot.subgraph(name='cluster_web') as web:
        web.attr(label='Web Layer', style='filled', color='lightblue')
        for controller in project['controllers']:
            web.node(controller, shape='box3d')

    with dot.subgraph(name='cluster_service') as service:
        service.attr(label='Service Layer', style='filled', color='lightgreen')
        for svc in project['services']:
            service.node(svc, shape='box')

    with dot.subgraph(name='cluster_data') as data:
        data.attr(label='Data Layer', style='filled', color='lightyellow')
        for repo in project['repositories']:
            data.node(repo, shape='cylinder')
        for entity in project['entities']:
            data.node(entity, shape='ellipse')

    # Add dependencies
    for dep in project['dependencies']:
        if 'spring' in dep.lower():
            dot.node(dep, shape='diamond', color='red')

    # Add Kafka topics
    if project['kafka_topics']:
        with dot.subgraph(name='cluster_kafka') as kafka:
            kafka.attr(label='Messaging', style='filled', color='lavender')
            for topic in project['kafka_topics']:
                kafka.node(topic, shape='note')

    # Add relationships
    for controller in project['controllers']:
        for service in project['services']:
            dot.edge(controller, service, label='calls')

    for service in project['services']:
        for repo in project['repositories']:
            dot.edge(service, repo, label='uses')
        for entity in project['entities']:
            dot.edge(service, entity, label='manages')

    for topic in project['kafka_topics']:
        if project['services']:
            dot.edge(project['services'][0], topic, label='publishes', style='dashed')

    return dot

# ... existing code ...

def main(repo_url: str):
    try:
        print("Cloning repository...")
        repo_path = clone_github_repo(repo_url)

        print("Analyzing projects...")
        projects = []
        
        # Define candidate source directories in priority order
        candidate_dirs = [
            os.path.join(repo_path, 'src', 'main', 'java'),
            os.path.join(repo_path, 'app', 'src', 'main', 'java'),
            repo_path  # Fallback to root
        ]

        # Find the first valid source directory
        src_dir = next((d for d in candidate_dirs if os.path.exists(d)), None)
        
        if not src_dir:
            print("No valid source directory found")
            return None

        project = analyze_java_project(src_dir)
        if project['files']:
            projects.append(project)
# def main(repo_url: str):
#     """Main execution flow with error handling"""
#     try:
#         print("Cloning repository...")
#         repo_path = clone_github_repo(repo_url)

#         print("Analyzing projects...")
#         projects = []
#         src_dirs = [
#             os.path.join(repo_path, 'src', 'main', 'java'),
#             os.path.join(repo_path, 'app', 'src', 'main', 'java'),
#             repo_path  # Fallback to root directory
#         ]

#         for src_dir in src_dirs:
#             if os.path.exists(src_dir):
#                 project = analyze_java_project(src_dir)
#                 if project['files']:
#                     projects.append(project)

        if not projects:
            print("No valid Java projects found")
            return None

        print("\nGenerating documentation...")
        for project in projects:
            project['summary'] = generate_project_summary(project)
            print(f"\nProject: {project['name']}")
            print("=" * 50)
            print(project['summary'])

        print("\nBuilding knowledge graph...")
        knowledge_graph = build_knowledge_graph(projects)
        visualize_knowledge_graph(knowledge_graph)

        print("\nInitializing RAG system...")
        rag = ArchitectureRAG(projects, knowledge_graph)

        print("\nGenerating architecture diagrams...")
        for project in projects:
            diagram = generate_architecture_diagram(project)
            try:
                diagram.render(f'architecture_{project["name"]}', cleanup=True)
                print(f"Generated diagram for {project['name']}")
            except Exception as e:
                print(f"Error generating diagram for {project['name']}: {str(e)}")

        print("\nAnalysis complete!")
        return rag

    except Exception as e:
        print(f"Critical error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
            
# Example usage
if __name__ == "__main__":
    # REPO_URL = "https://github.com/spring-projects/spring-petclinic"
    REPO_URL = "https://github.com/kishanrajput23/Java-Projects-Collections.git"
    rag_system = main(REPO_URL)

    if rag_system:
        # Example queries
        queries = [
            "How do the services communicate with each other?",
            "What are the main components of this system?",
            "How is data persistence implemented?",
            "What is the business domain of this application?",
            "Show me the REST API structure"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            try:
                result = rag_system.query(query)
                print(f"Answer: {result['answer']}")
                print(f"Sources: {result['context_sources']}")
            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")
    else:
        print("Analysis failed")
