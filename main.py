import os
from dotenv import load_dotenv
load_dotenv()
import tempfile
import shutil
import hashlib
import re
import ast
import json
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path
import javalang
from javalang.tree import (ClassDeclaration, MethodDeclaration,
                          FieldDeclaration, Annotation,
                          PackageDeclaration, Import)
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

# Directories to skip during analysis
SKIP_DIRS = {
    "test", "target", "build", "node_modules", ".git",
    "resources-noncompilable", "xdocs-examples", "out", "bin",
    "generated", "docs", "examples", "lib", "logs"
}

def should_skip_file(file_path: str) -> bool:
    """Determine if a file should be skipped based on its path"""
    path_parts = Path(file_path).parts
    return any(part in SKIP_DIRS for part in path_parts)

def detect_java_version(project_path: str) -> int:
    """Detect the Java version from build files with improved pattern matching"""
    build_files = {
        "build.gradle": [
            r"java\.toolchain\.languageVersion\s*=\s*JavaLanguageVersion\.of\((\d+)\)",
            r"sourceCompatibility\s*=\s*JavaVersion\.VERSION_(\d+)",
            r"targetCompatibility\s*=\s*['\"](\d+)['\"]"
        ],
        "pom.xml": [
            r"<maven\.compiler\.(?:source|target)>(\d+)<",
            r"<java\.version>(\d+)<"
        ],
        "build.gradle.kts": [
            r"java\.toolchain\.languageVersion\.set\(JavaLanguageVersion\.of\((\d+)\)\)",
            r"targetCompatibility\s*=\s*JavaVersion\.VERSION_(\d+)"
        ]
    }

    for build_file, patterns in build_files.items():
        path = os.path.join(project_path, build_file)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                for pattern in patterns:
                    match = re.search(pattern, content)
                    if match:
                        version = match.group(1)
                        # Handle version strings like "1.8" or "11"
                        if version.startswith('1.'):
                            return int(version.split('.')[1])
                        return int(version)
    return 8  # Default to Java 8 if not specified

from javalang.tree import CompilationUnit  # optional typing clarity

def safe_parse_java(content: str) -> Optional[CompilationUnit]:

    """Safely parse Java content with error recovery"""
    try:
        return javalang.parse.parse(content)
    except (javalang.parser.JavaSyntaxError, TypeError) as e:
        print(f"Parse error: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected parse error: {str(e)}")
        return None

def parse_java_file(file_path: str) -> Dict:
    """Robust Java parser that handles edge cases"""
    if should_skip_file(file_path):
        return empty_parse_result()

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        except Exception as e:
            print(f"Failed to read {file_path}: {str(e)}")
            return empty_parse_result()

    tree = safe_parse_java(content)
    if not tree:
        return empty_parse_result()

    package = tree.package.name if tree.package else ""

    # Safely extract imports
    imports = []
    if hasattr(tree, 'imports') and tree.imports:
        for imp in tree.imports:
            if isinstance(imp, Import):
                imports.append(imp.path)

    classes = []
    methods = []
    annotations = []
    fields = []
    rest_endpoints = []

    # Extract detailed class information with robust traversal
    for _, node in tree.filter(include_primitives=False):
        if isinstance(node, ClassDeclaration):
            class_info = {
                "name": node.name,
                "type": "class",
                "modifiers": getattr(node, 'modifiers', []) or [],
                "annotations": [ann.name for ann in getattr(node, 'annotations', []) or []],
                "methods": [],
                "fields": [],
                "extends": getattr(node.extends, 'name', None) if hasattr(node, 'extends') and node.extends else None,
                "implements": [impl.name for impl in node.implements] if hasattr(node, 'implements') and node.implements else []
            }

            # Safely process class body
            if hasattr(node, 'body') and node.body:
                for body_item in node.body:
                    if isinstance(body_item, MethodDeclaration):
                        method_info = extract_method_info(body_item)
                        class_info["methods"].append(method_info)
                        methods.append(body_item.name)

                        # Extract REST endpoints
                        for ann in method_info["annotations"]:
                            if "Mapping" in ann:
                                rest_endpoints.append({
                                    "class": node.name,
                                    "method": method_info["name"],
                                    "annotation": ann,
                                    "url": ann.replace("Mapping", "").lower() + "_path"
                                })

                    elif isinstance(body_item, FieldDeclaration):
                        field_declarators = getattr(body_item, 'declarators', []) or []
                        for declarator in field_declarators:
                            field_info = {
                                "name": getattr(declarator, 'name', ''),
                                "type": getattr(body_item.type, 'name', '') if hasattr(body_item, 'type') else '',
                                "modifiers": getattr(body_item, 'modifiers', []) or [],
                                "annotations": [ann.name for ann in getattr(body_item, 'annotations', []) or []]
                            }
                            if field_info["name"]:  # Only add if we have a valid name
                                class_info["fields"].append(field_info)
                                fields.append(field_info["name"])

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

def empty_parse_result() -> Dict:
    """Return an empty parse result dictionary"""
    return {
        "package": "",
        "classes": [],
        "imports": [],
        "methods": [],
        "annotations": [],
        "fields": [],
        "rest_endpoints": []
    }

def extract_method_info(method_node: MethodDeclaration) -> Dict:
    """Extract method information with robust attribute checking"""
    return {
        "name": getattr(method_node, 'name', ''),
        "return_type": getattr(method_node.return_type, 'name', 'void') if hasattr(method_node, 'return_type') and method_node.return_type else "void",
        "modifiers": getattr(method_node, 'modifiers', []) or [],
        "annotations": [ann.name for ann in getattr(method_node, 'annotations', []) or []],
        "parameters": [
            {
                "type": getattr(param.type, 'name', '') if hasattr(param, 'type') else '',
                "name": getattr(param, 'name', '')
            }
            for param in getattr(method_node, 'parameters', []) or []
            if hasattr(param, 'type') and hasattr(param, 'name')
        ]
    }

def clone_github_repo(repo_url: str) -> str:
    """Clone GitHub repository with improved error handling and validation"""
    try:
        if not repo_url.startswith(('https://github.com/', 'git@github.com:')):
            raise ValueError("Invalid GitHub URL format")

        # Extract owner and repo name more robustly
        repo_part = repo_url.split('github.com/')[-1].replace('.git', '')
        parts = repo_part.split('/')
        if len(parts) < 2:
            raise ValueError("Invalid GitHub URL - could not extract owner/repo")

        owner, repo_name = parts[0], parts[1]
        temp_dir = tempfile.mkdtemp(prefix=f"repo_{owner}_{repo_name}_")

        print(f"Cloning repository: {repo_url} to {temp_dir}")
        result = os.system(f"git clone --depth 1 {repo_url} {temp_dir}")

        if result != 0:
            raise RuntimeError(f"Git clone failed with exit code {result}")

        return temp_dir
    except Exception as e:
        print(f"Error cloning repository: {str(e)}")
        raise

def is_spring_project(project_path: str) -> bool:
    """Check if project is a Spring Boot project with more thorough checks"""
    # Check for build files
    build_files = ["pom.xml", "build.gradle", "build.gradle.kts"]
    for file in build_files:
        file_path = os.path.join(project_path, file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(spring_keyword in content
                           for spring_keyword in ["org.springframework.boot", "spring-boot-starter"]):
                        return True
            except Exception:
                continue

    # Check for Spring application class with more robust scanning
    spring_annotations = ["@SpringBootApplication", "@EnableAutoConfiguration"]
    for root, dirs, files in os.walk(project_path):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                if should_skip_file(file_path):
                    continue

                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        if any(ann in content for ann in spring_annotations):
                            return True
                except Exception:
                    continue
    return False

def analyze_java_project(project_path: str) -> Dict:
    """Comprehensive project analysis with improved error handling"""
    project_name = os.path.basename(project_path.rstrip('/'))
    is_spring = is_spring_project(project_path)
    java_version = detect_java_version(project_path)

    if java_version > 11:
        print(f"Note: Project uses Java {java_version} - modern features may not be fully parsed")

    project_structure = {
        "name": project_name,
        "path": project_path,
        "is_spring": is_spring,
        "java_version": java_version,
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
        "class_relationships": {"inheritance": [], "implementation": []},
        "metrics": {
            "total_files": 0,
            "parsed_files": 0,
            "error_files": 0
        }
    }

    # Extended domain recognition patterns
    domain_keywords = {
        "pet": "Pet Care/Veterinary",
        "customer": "CRM/Retail",
        "order": "E-commerce",
        "inventory": "Supply Chain",
        "financial": "Banking/Finance",
        "medical": "Healthcare",
        "clinic": "Healthcare",
        "owner": "CRM",
        "checkstyle": "Code Quality/Static Analysis",
        "style": "Code Quality",
        "lint": "Code Quality",
        "analysis": "Static Analysis",
        "audit": "Compliance",
        "rule": "Rules Engine"
    }

    # Walk through the project directory with better directory skipping
    for root, dirs, files in os.walk(project_path):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

        for file in files:
            if not file.endswith(".java"):
                continue

            project_structure["metrics"]["total_files"] += 1
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, project_path)

            if should_skip_file(file_path):
                continue

            try:
                parsed_data = parse_java_file(file_path)
                project_structure["metrics"]["parsed_files"] += 1

                # Business domain detection with content sampling
                if not project_structure["business_domain"]:
                    try:
                        with open(file_path, 'r', errors='ignore') as f:
                            content = f.read(4096)  # Only read first 4KB for efficiency
                            for keyword, domain in domain_keywords.items():
                                if re.search(rf'\b{keyword}\b', content, re.IGNORECASE):
                                    project_structure["business_domain"] = domain
                                    break
                    except Exception:
                        pass

                # Classify components with null checks
                for class_info in parsed_data.get("classes", []):
                    class_name = class_info.get("name")
                    if not class_name:
                        continue

                    project_structure["classes"].append(class_name)

                    # Spring component detection
                    if is_spring:
                        for ann in class_info.get("annotations", []):
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
                    for ann in class_info.get("annotations", []):
                        if ann in KAFKA_ANNOTATIONS:
                            project_structure["kafka_topics"].add(f"{class_name}_Topic")

                    # Class relationships with validation
                    extends = class_info.get("extends")
                    if extends:
                        project_structure["class_relationships"]["inheritance"].append(
                            (class_name, extends)
                        )

                    for interface in class_info.get("implements", []):
                        project_structure["class_relationships"]["implementation"].append(
                            (class_name, interface)
                        )

                # REST endpoints
                project_structure["rest_endpoints"].extend(parsed_data.get("rest_endpoints", []))

                # Update project structure with file info
                file_info = {
                    "path": relative_path,
                    "package": parsed_data.get("package", ""),
                    "classes": [c.get("name") for c in parsed_data.get("classes", []) if c.get("name")],
                    "imports": parsed_data.get("imports", []),
                    "methods": parsed_data.get("methods", []),
                    "annotations": parsed_data.get("annotations", []),
                    "fields": parsed_data.get("fields", [])
                }
                project_structure["files"].append(file_info)

                # Update dependencies with filtering
                imports = parsed_data.get("imports", [])
                filtered_imports = [imp for imp in imports if not imp.startswith('java.')]
                project_structure["dependencies"].update(filtered_imports)

            except Exception as e:
                project_structure["metrics"]["error_files"] += 1
                print(f"Error processing {file_path}: {str(e)}")

    # Post-processing to clean up data
    project_structure["dependencies"] = sorted(project_structure["dependencies"])
    project_structure["kafka_topics"] = sorted(project_structure["kafka_topics"])

    return project_structure

def generate_project_summary(project: Dict) -> str:
    """Generate detailed project summary using Gemini with fallback"""
    prompt = f"""
    Analyze this Java project structure and generate a comprehensive summary:

    Project Name: {project.get('name', 'Unknown')}
    Business Domain: {project.get('business_domain', 'Not identified')}
    Spring Project: {'Yes' if project.get('is_spring', False) else 'No'}
    Java Version: {project.get('java_version', 8)}

    Key Metrics:
    - Files: {project.get('metrics', {}).get('total_files', 0)} total
    - Parsed: {project.get('metrics', {}).get('parsed_files', 0)} successfully
    - Errors: {project.get('metrics', {}).get('error_files', 0)}

    Key Components:
    - Controllers: {len(project.get('controllers', []))}
    - Services: {len(project.get('services', []))}
    - Repositories: {len(project.get('repositories', []))}
    - Entities: {len(project.get('entities', []))}

    REST Endpoints: {len(project.get('rest_endpoints', []))}
    Kafka Topics: {len(project.get('kafka_topics', []))}
    Dependencies: {len(project.get('dependencies', []))}

    Please provide:
    1. Business domain analysis based on the components
    2. Architectural pattern recognition (MVC, Microservices, etc.)
    3. Technology stack identification
    4. Data flow between key components
    5. Deployment characteristics
    6. Code quality observations based on the structure
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback summary
        return f"""Project Analysis Summary:
Name: {project.get('name', 'Unknown')}
Domain: {project.get('business_domain', 'Not identified')}
Type: {'Spring' if project.get('is_spring', False) else 'Standard Java'}
Java Version: {project.get('java_version', 8)}

Key Components:
- Classes: {len(project.get('classes', []))}
- Controllers: {len(project.get('controllers', []))}
- Services: {len(project.get('services', []))}
- Repositories: {len(project.get('repositories', []))}
- Entities: {len(project.get('entities', []))}

Integration Points:
- REST Endpoints: {len(project.get('rest_endpoints', []))}
- Kafka Topics: {len(project.get('kafka_topics', []))}
- External Dependencies: {len(project.get('dependencies', []))}

Analysis unavailable due to: {str(e)}
"""

def build_knowledge_graph(projects: List[Dict]) -> nx.DiGraph:
    """Build a comprehensive knowledge graph with improved relationship handling"""
    G = nx.DiGraph()

    for project in projects:
        # Add project node with enhanced attributes
        G.add_node(project['name'],
                  type='project',
                  size=len(project.get('files', [])),
                  spring=project.get('is_spring', False),
                  domain=project.get('business_domain', ''),
                  java_version=project.get('java_version', 8))

        # Add classes with types and validation
        for cls in project.get('classes', []):
            if not cls:
                continue

            node_type = 'class'
            if cls in project.get('controllers', []):
                node_type = 'controller'
            elif cls in project.get('services', []):
                node_type = 'service'
            elif cls in project.get('repositories', []):
                node_type = 'repository'
            elif cls in project.get('entities', []):
                node_type = 'entity'

            G.add_node(cls, type=node_type)
            G.add_edge(project['name'], cls, relationship='contains')

        # Add dependencies with filtering
        for dep in project.get('dependencies', []):
            if not dep:
                continue
            G.add_node(dep, type='dependency')
            G.add_edge(project['name'], dep, relationship='depends_on')

        # Add REST endpoints with validation
        for endpoint in project.get('rest_endpoints', []):
            if not endpoint.get('class') or not endpoint.get('method'):
                continue
            endpoint_id = f"{endpoint['class']}.{endpoint['method']}"
            G.add_node(endpoint_id,
                      type='endpoint',
                      method=endpoint.get('annotation', 'unknown'))
            G.add_edge(endpoint['class'], endpoint_id, relationship='exposes')

        # Add Kafka topics
        for topic in project.get('kafka_topics', []):
            if not topic:
                continue
            G.add_node(topic, type='kafka_topic')
            G.add_edge(project['name'], topic, relationship='uses_messaging')

        # Add class relationships with validation
        for rel_type, relationships in project.get('class_relationships', {}).items():
            for relationship in relationships:
                if (isinstance(relationship, (list, tuple)) and len(relationship) == 2):
                    source, target = relationship
                    if source and target and source in project.get('classes', []):
                        G.add_edge(source, target, relationship=rel_type)

    # Connect projects through shared dependencies with threshold
    for i, proj1 in enumerate(projects):
        for j, proj2 in enumerate(projects[i+1:], i+1):
            common_deps = set(proj1.get('dependencies', [])) & set(proj2.get('dependencies', []))
            if len(common_deps) >= 3:  # Only connect if they share at least 3 dependencies
                for dep in common_deps:
                    G.add_edge(proj1['name'], proj2['name'],
                              relationship='shares_dependency',
                              dependency=dep,
                              weight=len(common_deps))

    return G

def visualize_knowledge_graph(graph: nx.DiGraph, filename: str = "knowledge_graph.html"):
    """Enhanced visualization with component types and layout improvements"""
    net = Network(notebook=True, height="900px", width="100%",
                 bgcolor="#222222", font_color="white",
                 directed=True)

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

    # Shape mapping
    type_shapes = {
        'project': 'dot',
        'controller': 'diamond',
        'service': 'square',
        'repository': 'triangle',
        'entity': 'ellipse',
        'class': 'box',
        'dependency': 'star',
        'endpoint': 'database',
        'kafka_topic': 'text'
    }

    # Add nodes with enhanced styling
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'class')
        color = type_colors.get(node_type, '#888888')
        shape = type_shapes.get(node_type, 'box')
        title = f"<b>{node}</b><br>Type: {node_type}"

        # Add additional attributes to tooltip
        if node_type == 'project':
            title += f"<br>Domain: {data.get('domain', '')}"
            title += f"<br>Java: {data.get('java_version', '?')}"
            title += f"<br>Spring: {'Yes' if data.get('spring', False) else 'No'}"
            size = 35
        elif node_type in ('controller', 'service'):
            size = 25
        elif node_type == 'dependency':
            size = 20
        else:
            size = 15

        net.add_node(node,
                    label=node,
                    color=color,
                    title=title,
                    size=size,
                    shape=shape,
                    borderWidth=2)

    # Add edges with enhanced styling
    for source, target, data in graph.edges(data=True):
        rel = data.get('relationship', 'related')
        color = '#FFFFFF'
        dashes = False
        width = 2

        if rel == 'contains':
            color = '#AAAAAA'
        elif rel == 'depends_on':
            color = '#FF6B6B'
            dashes = True
            width = 1
        elif rel == 'exposes':
            color = '#4ECDC4'
            width = 3
        elif rel == 'uses_messaging':
            color = '#F15BB5'
            width = 3
        elif rel == 'shares_dependency':
            color = '#FFA500'
            width = 1 + (data.get('weight', 1) / 2)

        net.add_edge(source, target,
                    title=rel,
                    color=color,
                    dashes=dashes,
                    width=width,
                    arrowStrikethrough=False)

    # Improved physics configuration
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {
          "size": 14,
          "face": "arial",
          "strokeWidth": 2,
          "strokeColor": "#000000"
        },
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.5)",
          "size": 10,
          "x": 5,
          "y": 5
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.8
          }
        },
        "smooth": {
          "type": "continuous",
          "roundness": 0.5
        },
        "selectionWidth": 3,
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.3)",
          "size": 5,
          "x": 3,
          "y": 3
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.5,
          "springLength": 200,
          "springConstant": 0.05,
          "damping": 0.2,
          "avoidOverlap": 0.2
        },
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 25
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": true,
        "multiselect": true
      }
    }
    """)

    try:
        net.show(filename, notebook=False)
        print(f"Knowledge graph saved to {filename}")
    except Exception as e:
        print(f"Error saving knowledge graph: {str(e)}")

class ArchitectureRAG:
    def __init__(self, projects: List[Dict], knowledge_graph: nx.DiGraph):
        self.projects = projects
        self.knowledge_graph = knowledge_graph
        self.vector_index = {}
        self._build_vector_index()

    def _build_vector_index(self):
        """Build a comprehensive vector index with metadata"""
        for project in self.projects:
            doc = generate_project_summary(project)
            doc_hash = hashlib.md5(doc.encode()).hexdigest()

            # Enhanced metadata
            self.vector_index[doc_hash] = {
                "content": doc,
                "metadata": {
                    "type": "project_summary",
                    "project": project.get('name', ''),
                    "domain": project.get('business_domain', ''),
                    "spring": project.get('is_spring', False),
                    "java_version": project.get('java_version', 8),
                    "components": {
                        "controllers": project.get('controllers', []),
                        "services": project.get('services', []),
                        "repositories": project.get('repositories', []),
                        "entities": project.get('entities', [])
                    },
                    "metrics": {
                        "files": len(project.get('files', [])),
                        "classes": len(project.get('classes', [])),
                        "endpoints": len(project.get('rest_endpoints', []))
                    }
                }
            }

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Enhanced query with context-aware response and fallback"""
        # Step 1: Use Gemini to understand the query with retry
        query_analysis = self._analyze_query(question)

        # Step 2: Find relevant context with scoring
        relevant_docs = self._find_relevant_docs(query_analysis, top_k)

        # Step 3: Generate comprehensive answer with fallback
        return self._generate_answer(question, relevant_docs)

    def _analyze_query(self, question: str) -> Dict:
        """Analyze the query with retry logic"""
        prompt = f"""
        Analyze this architecture question and extract key elements:

        Question: {question}

        Identify:
        1. Project components mentioned (controllers, services, etc.)
        2. Architectural concerns (scaling, security, data flow)
        3. Technology references (Spring, Kafka, databases)
        4. Business domain aspects
        5. Specific patterns or practices mentioned

        Return your analysis as a well-formatted JSON object with keys:
        components, concerns, technologies, domain, patterns
        """

        try:
            response = model.generate_content(prompt)
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback analysis
                components = re.findall(r'\b(controller|service|repository|entity)\b', question, re.IGNORECASE)
                technologies = re.findall(r'\b(spring|kafka|jpa|hibernate)\b', question, re.IGNORECASE)
                return {
                    "components": components,
                    "technologies": technologies,
                    "concerns": [],
                    "domain": "",
                    "patterns": []
                }
        except Exception as e:
            print(f"Error analyzing query: {str(e)}")
            return {
                "components": [],
                "concerns": [],
                "technologies": [],
                "domain": "",
                "patterns": []
            }

    def _find_relevant_docs(self, query_analysis: Dict, top_k: int) -> List[Dict]:
        """Find relevant documents with scoring"""
        relevant_docs = []

        for doc_data in self.vector_index.values():
            doc_text = doc_data["content"].lower()
            metadata = doc_data["metadata"]
            match_score = 0

            # Component matching
            for comp in query_analysis.get("components", []):
                if comp.lower() in doc_text:
                    match_score += 2
                # Also check metadata
                if any(comp.lower() in c.lower()
                      for c in metadata["components"].get("controllers", [])):
                    match_score += 1
                if any(comp.lower() in c.lower()
                      for c in metadata["components"].get("services", [])):
                    match_score += 1

            # Technology matching
            for tech in query_analysis.get("technologies", []):
                if tech.lower() in doc_text:
                    match_score += 2
                if tech.lower() == "spring" and metadata["spring"]:
                    match_score += 3

            # Domain matching
            domain = query_analysis.get("domain", "")
            if domain and domain.lower() in doc_text:
                match_score += 3
            if domain and domain.lower() in metadata["domain"].lower():
                match_score += 2

            # Pattern matching
            for pattern in query_analysis.get("patterns", []):
                if pattern.lower() in doc_text:
                    match_score += 1

            if match_score > 0:
                relevant_docs.append((match_score, doc_data))

        # Sort by relevance and take top_k
        relevant_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc[1] for doc in relevant_docs[:top_k]]

    def _generate_answer(self, question: str, relevant_docs: List[Dict]) -> Dict:
        """Generate answer with context and fallback"""
        if not relevant_docs:
            return {
                "answer": "No relevant information found in the project documentation.",
                "context_sources": []
            }

        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        sources = [doc["metadata"]["project"] for doc in relevant_docs]

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
        6. Recommendations if applicable

        Format your response with clear section headings.
        """

        try:
            response = model.generate_content(prompt)
            return {
                "answer": response.text,
                "context_sources": sources
            }
        except Exception as e:
            # Fallback answer
            return {
                "answer": f"""I couldn't generate a complete answer due to an error: {str(e)}

Based on the context from {', '.join(sources)}, here's what I can share:
- Relevant projects: {', '.join(sources)}
- Technologies mentioned: {', '.join(set(doc['metadata']['technologies'] for doc in relevant_docs))}
""",
                "context_sources": sources
            }

def generate_architecture_diagram(project: Dict) -> Digraph:
    """Generate detailed architecture diagram with improved layout"""
    dot = Digraph(comment=project.get('name', 'project'),
                 format='png',
                 graph_attr={
                     'rankdir': 'TB',
                     'labelloc': 't',
                     'fontname': 'Helvetica',
                     'fontsize': '16',
                     'fontcolor': 'black'
                 },
                 node_attr={
                     'fontname': 'Helvetica',
                     'fontsize': '12'
                 })

    project_name = project.get('name', 'Unknown Project')
    dot.attr(label=f"{project_name} Architecture\n({project.get('business_domain', '')})")

    # Cluster for layers with improved styling
    with dot.subgraph(name='cluster_web') as web:
        web.attr(label='Web Layer',
                style='filled,rounded',
                color='lightblue',
                fontname='Helvetica',
                fontsize='14')
        for controller in project.get('controllers', []):
            web.node(controller,
                    shape='box3d',
                    style='filled',
                    color='#4ECDC4')

    with dot.subgraph(name='cluster_service') as service:
        service.attr(label='Service Layer',
                    style='filled,rounded',
                    color='lightgreen',
                    fontname='Helvetica',
                    fontsize='14')
        for svc in project.get('services', []):
            service.node(svc,
                       shape='box',
                       style='filled',
                       color='#FFD166')

    with dot.subgraph(name='cluster_data') as data:
        data.attr(label='Data Layer',
                 style='filled,rounded',
                 color='lightyellow',
                 fontname='Helvetica',
                 fontsize='14')
        for repo in project.get('repositories', []):
            data.node(repo,
                    shape='cylinder',
                    style='filled',
                    color='#06D6A0')
        for entity in project.get('entities', []):
            data.node(entity,
                     shape='ellipse',
                     style='filled',
                     color='#118AB2')

    # Add dependencies with filtering
    important_deps = [dep for dep in project.get('dependencies', [])
                     if any(kw in dep.lower()
                            for kw in ['spring', 'hibernate', 'kafka', 'jpa'])]

    with dot.subgraph(name='cluster_deps') as deps:
        deps.attr(label='Key Dependencies',
                 style='filled,rounded',
                 color='lavender')
        for dep in important_deps:
            deps.node(dep,
                     shape='diamond',
                     style='filled',
                     color='#EF476F')

    # Add Kafka topics if they exist
    if project.get('kafka_topics'):
        with dot.subgraph(name='cluster_kafka') as kafka:
            kafka.attr(label='Messaging',
                      style='filled,rounded',
                      color='#F15BB5',
                      fontcolor='white')
            for topic in project.get('kafka_topics', []):
                kafka.node(topic,
                          shape='note',
                          style='filled',
                          color='#F15BB5',
                          fontcolor='white')

    # Add relationships with improved styling
    for controller in project.get('controllers', []):
        for service in project.get('services', []):
            dot.edge(controller, service,
                    label='calls',
                    style='solid',
                    color='#4ECDC4')

    for service in project.get('services', []):
        for repo in project.get('repositories', []):
            dot.edge(service, repo,
                    label='uses',
                    style='solid',
                    color='#06D6A0')
        for entity in project.get('entities', []):
            dot.edge(service, entity,
                    label='manages',
                    style='solid',
                    color='#118AB2')

    # Add messaging relationships if they exist
    if project.get('kafka_topics') and project.get('services'):
        primary_service = project['services'][0]
        for topic in project['kafka_topics']:
            dot.edge(primary_service, topic,
                    label='publishes',
                    style='dashed',
                    color='#F15BB5')

    # Add dependency relationships
    for dep in important_deps:
        if project.get('controllers'):
            dot.edge(project['controllers'][0], dep,
                    style='dotted',
                    color='#888888')
        if project.get('repositories'):
            dot.edge(project['repositories'][0], dep,
                    style='dotted',
                    color='#888888')

    return dot

def main(repo_url: str, analyze_deps: bool = True) -> Optional[ArchitectureRAG]:
    """Enhanced main execution flow with dependency analysis option"""
    try:
        print("Cloning repository...")
        repo_path = clone_github_repo(repo_url)

        print("\nAnalyzing projects...")
        projects = []

        # Potential source directories to check
        potential_src_dirs = [
            os.path.join(repo_path, 'src', 'main', 'java'),
            os.path.join(repo_path, 'app', 'src', 'main', 'java'),
            os.path.join(repo_path, 'main', 'java'),
            repo_path  # Fallback to root directory
        ]

        # Find the first existing source directory
        src_dir = next((d for d in potential_src_dirs if os.path.exists(d)), None)

        if not src_dir:
            print("No Java source directory found in expected locations")
            return None

        project = analyze_java_project(src_dir)
        if not project.get('files'):
            print("No valid Java files found in the project")
            return None

        projects.append(project)

        print("\nGenerating documentation...")
        for proj in projects:
            proj['summary'] = generate_project_summary(proj)
            print(f"\nProject: {proj['name']}")
            print("=" * 50)
            print(proj['summary'])

        print("\nBuilding knowledge graph...")
        knowledge_graph = build_knowledge_graph(projects)
        visualize_knowledge_graph(knowledge_graph, f"{project['name']}_knowledge_graph.html")

        print("\nInitializing RAG system...")
        rag = ArchitectureRAG(projects, knowledge_graph)

        print("\nGenerating architecture diagrams...")
        for proj in projects:
            try:
                diagram = generate_architecture_diagram(proj)
                output_file = f"architecture_{proj['name']}"
                diagram.render(output_file, cleanup=True, format='png')
                print(f"Generated diagram: {output_file}.png")
            except Exception as e:
                print(f"Error generating diagram: {str(e)}")

        print("\nAnalysis complete!")
        return rag

    except Exception as e:
        print(f"Critical error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up cloned repository
        if 'repo_path' in locals() and os.path.exists(repo_path):
            try:
                shutil.rmtree(repo_path)
                print(f"Cleaned up temporary repository at {repo_path}")
            except Exception as e:
                print(f"Error cleaning up repository: {str(e)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Java project architecture')
    parser.add_argument('repo_url', type=str, help='GitHub repository URL to analyze')
    args = parser.parse_args()

    rag_system = main(args.repo_url)

    if rag_system:
        example_queries = [
            "How do the components in this system communicate with each other?",
            "What architectural patterns are used in this project?",
            "How is data persistence implemented?",
            "What is the primary business domain of this application?",
            "Describe the REST API structure if it exists",
            "Are there any messaging systems like Kafka used?",
            "What are the key dependencies of this project?",
            "How would you improve the architecture of this system?"
        ]

        print("\nYou can now ask questions about the project architecture.")
        print("Here are some example questions you might want to try:\n")
        for i, query in enumerate(example_queries, 1):
            print(f"{i}. {query}")

        while True:
            try:
                question = input("\nEnter your question (or 'quit' to exit): ")
                if question.lower() in ('quit', 'exit'):
                    break

                if not question.strip():
                    continue

                result = rag_system.query(question)
                print(f"\nAnswer:\n{result['answer']}")
                if result['context_sources']:
                    print(f"\nSources: {', '.join(result['context_sources'])}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing query: {str(e)}")
    else:
        print("Analysis failed")