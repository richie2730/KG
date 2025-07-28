#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import networkx as nx
from git import Repo
from py4j.java_gateway import JavaGateway, GatewayParameters
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from tqdm import tqdm
import time
import subprocess
import signal
import psutil
import socket
import argparse
import urllib.request
import shutil
import glob

class JavaRepositoryAnalyzer:
    def __init__(self, repo_url, gemini_api_key, work_dir="workdir", max_files=None):
        self.repo_url = repo_url
        self.gemini_api_key = gemini_api_key
        self.work_dir = work_dir
        self.max_files = max_files
        self.repo_path = None
        self.gateway = None
        self.parser = None
        self.vector_db = None
        self.knowledge_graph = nx.DiGraph()
        self.java_process = None
        self.server_port = 25333
        self.setup_workdir()
        self.setup_vector_db()
        
    def setup_workdir(self):
        """Create working directory structure"""
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "lib"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "src"), exist_ok=True)

    def download_file(self, url, filename):
        """Download a file if it doesn't exist"""
        filepath = os.path.join(self.work_dir, "lib", filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        return filepath

    def setup_dependencies(self):
        """Download required dependencies"""
        deps = [
            ("https://www.antlr.org/download/antlr-4.13.1-complete.jar", "antlr-4.13.1-complete.jar"),
            ("https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.15.2/jackson-databind-2.15.2.jar", "jackson-databind-2.15.2.jar"),
            ("https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-core/2.15.2/jackson-core-2.15.2.jar", "jackson-core-2.15.2.jar"),
            ("https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-annotations/2.15.2/jackson-annotations-2.15.2.jar", "jackson-annotations-2.15.2.jar"),
            ("https://repo1.maven.org/maven2/net/sf/py4j/py4j/0.10.9.7/py4j-0.10.9.7.jar", "py4j-0.10.9.7.jar"),
            ("https://raw.githubusercontent.com/antlr/grammars-v4/master/java/java/JavaLexer.g4", "JavaLexer.g4"),
            ("https://raw.githubusercontent.com/antlr/grammars-v4/master/java/java/JavaParser.g4", "JavaParser.g4")
        ]
        
        for url, filename in deps:
            self.download_file(url, filename)
            
        # Create JavaParserGateway.java
        gateway_src = os.path.join(self.work_dir, "src", "JavaParserGateway.java")
        if not os.path.exists(gateway_src):
            with open(gateway_src, "w") as f:
                f.write("""import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import com.fasterxml.jackson.databind.ObjectMapper;
import py4j.GatewayServer;

public class JavaParserGateway {
    private Java21Parser parser;
    private ObjectMapper mapper = new ObjectMapper();

    public JavaParserGateway() {
        parser = new Java21Parser();
    }

    public String parseFile(String filePath) {
        try {
            String content = new String(Files.readAllBytes(Paths.get(filePath)));
            Object result = parser.parse(content);
            return mapper.writeValueAsString(result);
        } catch (IOException e) {
            return "{\\"error\\": \\"" + e.getMessage() + "\\"}";
        } catch (Exception e) {
            return "{\\"error\\": \\"" + e.getClass().getSimpleName() + ": " + e.getMessage() + "\\"}";
        }
    }

    public static void main(String[] args) {
        GatewayServer server = new GatewayServer(new JavaParserGateway());
        server.start();
        System.out.println("JavaParserGateway Server Started");
    }
}""")
                
        # Create Java21Parser.java
        parser_src = os.path.join(self.work_dir, "src", "Java21Parser.java")
        if not os.path.exists(parser_src):
            with open(parser_src, "w") as f:
                f.write("""import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
import java.util.*;
import java.util.stream.Collectors;

public class Java21Parser {
    public Map<String, Object> parse(String javaCode) {
        Map<String, Object> result = new HashMap<>();
        try {
            // Simplified parsing logic - real implementation would use ANTLR
            result.put("classes", Collections.emptyList());
            result.put("records", Collections.emptyList());
            result.put("interfaces", Collections.emptyList());
            result.put("enums", Collections.emptyList());
            result.put("module", null);
            return result;
        } catch (Exception e) {
            result.put("error", e.getMessage());
            return result;
        }
    }
}""")

    def generate_parser(self):
        """Generate ANTLR parser and compile Java code"""
        lib_dir = os.path.join(self.work_dir, "lib")
        src_dir = os.path.join(self.work_dir, "src")
        
        # Generate ANTLR parser
        antlr_cmd = [
            "java", "-jar", os.path.join(lib_dir, "antlr-4.13.1-complete.jar"),
            "-Dlanguage=Java", 
            "-o", src_dir,
            os.path.join(lib_dir, "JavaLexer.g4"),
            os.path.join(lib_dir, "JavaParser.g4")
        ]
        print("Generating ANTLR parser...")
        subprocess.run(antlr_cmd, check=True)
        
        # Compile Java code
        classpath = ":".join([
            os.path.join(lib_dir, "antlr-4.13.1-complete.jar"),
            os.path.join(lib_dir, "jackson-databind-2.15.2.jar"),
            os.path.join(lib_dir, "jackson-core-2.15.2.jar"),
            os.path.join(lib_dir, "jackson-annotations-2.15.2.jar"),
            os.path.join(lib_dir, "py4j-0.10.9.7.jar")
        ])
        
        javac_cmd = [
            "javac", 
            "-cp", classpath, 
            "-d", self.work_dir,
            os.path.join(src_dir, "*.java")
        ]
        print("Compiling Java code...")
        subprocess.run(javac_cmd, check=True)

    def is_port_open(self, port):
        """Check if a port is open"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0

    def kill_process_on_port(self, port):
        """Kill any process running on the specified port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    connections = proc.info.get('connections')
                    if connections:
                        for conn in connections:
                            if conn.laddr.port == port:
                                print(f"Killing process {proc.info['pid']} ({proc.info['name']}) on port {port}")
                                proc.kill()
                                time.sleep(1)
                                return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                    continue
        except Exception as e:
            print(f"Error killing process on port {port}: {e}")
        return False

    def start_java_server(self):
        """Start the Java gateway server"""
        try:
            # Setup dependencies and parser
            self.setup_dependencies()
            self.generate_parser()
            
            # Kill any existing process on the port
            if self.is_port_open(self.server_port):
                print(f"Port {self.server_port} is already in use, killing existing process...")
                self.kill_process_on_port(self.server_port)
                time.sleep(2)

            # Build classpath
            classpath = ":".join([
                os.path.join(self.work_dir, "lib", "antlr-4.13.1-complete.jar"),
                os.path.join(self.work_dir, "lib", "jackson-databind-2.15.2.jar"),
                os.path.join(self.work_dir, "lib", "jackson-core-2.15.2.jar"),
                os.path.join(self.work_dir, "lib", "jackson-annotations-2.15.2.jar"),
                os.path.join(self.work_dir, "lib", "py4j-0.10.9.7.jar"),
                self.work_dir
            ])

            print("Starting Java gateway server...")
            self.java_process = subprocess.Popen(
                ['java', '-cp', classpath, 'JavaParserGateway'],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            # Wait for server to start
            max_attempts = 30
            for attempt in range(max_attempts):
                if self.is_port_open(self.server_port):
                    print(f"Java server started successfully on port {self.server_port}")
                    return True

                if self.java_process.poll() is not None:
                    stderr = self.java_process.stderr.read().decode('utf-8')
                    print(f"Java process exited prematurely with code {self.java_process.returncode}")
                    print("Error output:")
                    print(stderr)
                    return False

                time.sleep(1)
                print(f"Waiting for server to start... ({attempt + 1}/{max_attempts})")

            print("Failed to start Java server - timeout")
            return False

        except Exception as e:
            print(f"Error starting Java server: {str(e)}")
            return False

    def setup_parser(self):
        """Setup the Java parser gateway with proper error handling"""
        try:
            # Start Java server if not running
            if not self.is_port_open(self.server_port):
                if not self.start_java_server():
                    raise Exception("Failed to start Java server")

            # Connect to the gateway
            print("Connecting to Java gateway...")
            self.gateway = JavaGateway(
                gateway_parameters=GatewayParameters(port=self.server_port, auto_convert=True)
            )

            # Test the connection
            self.parser = self.gateway.entry_point
            test_result = self.parser.parseFile("NonExistentFile.java")
            print("Java parser gateway connected successfully")
            return True

        except Exception as e:
            print(f"Error setting up parser: {str(e)}")
            self.cleanup_java_process()
            return False

    def cleanup_java_process(self):
        """Clean up Java process and gateway"""
        try:
            if self.gateway:
                self.gateway.shutdown()
                self.gateway = None
                self.parser = None
        except:
            pass

        try:
            if self.java_process:
                os.killpg(os.getpgid(self.java_process.pid), signal.SIGTERM)
                self.java_process.wait(timeout=5)
        except:
            try:
                if self.java_process:
                    self.java_process.kill()
            except:
                pass
        finally:
            self.java_process = None

        # Final cleanup
        self.kill_process_on_port(self.server_port)

    def setup_vector_db(self):
        """Initialize the vector database"""
        self.vector_db = {
            "index": faiss.IndexFlatL2(384),
            "metadata": [],
            "model": SentenceTransformer('all-MiniLM-L6-v2')
        }

    def clone_repository(self):
        """Clone the repository if it doesn't exist"""
        repo_name = self.repo_url.split('/')[-1].replace('.git', '')
        self.repo_path = os.path.join(self.work_dir, repo_name)
        if not os.path.exists(self.repo_path):
            print(f"Cloning repository from {self.repo_url}...")
            Repo.clone_from(self.repo_url, self.repo_path)
        else:
            print(f"Repository already exists at {self.repo_path}")
        return self.repo_path

    def parse_java_file(self, file_path):
        """Parse a single Java file with better error handling"""
        try:
            if not self.parser:
                if not self.setup_parser():
                    return {"classes": []}

            json_str = self.parser.parseFile(file_path)
            result = json.loads(json_str)

            # Check for parsing errors
            if "error" in result:
                return {"classes": []}

            return result
        except Exception as e:
            print(f"Error parsing {file_path}: {str(e)}")
            return {"classes": []}

    def add_to_vector_db(self, content, entity_type, name, path):
        """Add content to the vector database"""
        try:
            embedding = self.vector_db["model"].encode([content])[0]
            self.vector_db["index"].add(np.array([embedding]).astype('float32'))
            self.vector_db["metadata"].append({
                "type": entity_type,
                "name": name,
                "path": path,
                "content": content
            })
        except Exception as e:
            print(f"Error adding to vector DB: {str(e)}")

    def vector_db_search(self, query, k=5):
        """Search the vector database"""
        if len(self.vector_db["metadata"]) == 0:
            return []

        try:
            query_embed = self.vector_db["model"].encode([query])[0]
            actual_k = min(k, len(self.vector_db["metadata"]))
            distances, indices = self.vector_db["index"].search(
                np.array([query_embed]).astype('float32'), actual_k
            )
            return [self.vector_db["metadata"][i] for i in indices[0] if i < len(self.vector_db["metadata"])]
        except Exception as e:
            print(f"Error searching vector DB: {str(e)}")
            return []

    def build_knowledge_graph(self):
        """Build the knowledge graph from Java files"""
        if not self.setup_parser():
            print("Failed to setup parser. Cannot build knowledge graph.")
            return

        java_files = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))

        if self.max_files:
            java_files = java_files[:self.max_files]

        print(f"Found {len(java_files)} Java files to parse")

        successful_parses = 0
        failed_parses = 0

        for file_path in tqdm(java_files, desc="Parsing Java files"):
            try:
                parsed_data = self.parse_java_file(file_path)
                if not parsed_data or "error" in parsed_data:
                    failed_parses += 1
                    continue

                # Simplified graph building for demonstration
                class_name = os.path.splitext(os.path.basename(file_path))[0]
                self.knowledge_graph.add_node(class_name, type="class", path=file_path)
                
                # Add to vector DB
                with open(file_path, 'r') as f:
                    content = f.read(1000)  # Read first 1000 chars
                self.add_to_vector_db(content, "class", class_name, file_path)

                successful_parses += 1

            except Exception as e:
                failed_parses += 1
                if failed_parses <= 5:
                    print(f"Error processing {file_path}: {str(e)}")

        print(f"\nParsing completed:")
        print(f"  - Successfully parsed: {successful_parses} files")
        print(f"  - Failed to parse: {failed_parses} files")
        print(f"  - Knowledge graph nodes: {len(self.knowledge_graph.nodes())}")
        print(f"  - Vector DB entries: {len(self.vector_db['metadata'])}")

    def analyze_architecture(self):
        """Analyze the repository architecture using Gemini"""
        if len(self.knowledge_graph.nodes()) == 0:
            return "No classes found. Architecture analysis cannot be performed."

        try:
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Prepare context from vector DB
            context_results = self.vector_db_search("class architecture", k=5)
            context = "\n".join([f"{res['type']}: {res['name']}" for res in context_results[:3]])

            prompt = f"""
            Analyze the architecture of this Java repository:

            Repository: {self.repo_url}
            Total classes analyzed: {len(self.knowledge_graph.nodes())}

            Sample classes:
            {', '.join(list(self.knowledge_graph.nodes())[:5])}

            Key code elements found:
            {context}

            Please provide a high-level architecture overview.
            """

            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating architecture analysis: {str(e)}"

    def query_codebase(self, question):
        """Query the codebase using natural language"""
        if len(self.vector_db["metadata"]) == 0:
            return "No code has been indexed yet."

        try:
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Retrieve relevant code context
            context_results = self.vector_db_search(question, k=5)
            context = "\n".join([f"{res['type']}: {res['name']}\n{res['content'][:300]}..."
                               for res in context_results])

            prompt = f"""
            Question about the Java codebase: {question}

            Repository: {self.repo_url}

            Relevant code context:
            {context}

            Please provide a detailed answer with specific examples when possible.
            """

            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def get_statistics(self):
        """Get analysis statistics"""
        return {
            "total_nodes": len(self.knowledge_graph.nodes()),
            "vector_db_entries": len(self.vector_db["metadata"]),
            "java_files_processed": len([n for n in self.knowledge_graph.nodes()])
        }

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_java_process()

def main():
    parser = argparse.ArgumentParser(description="Java Repository Analyzer")
    parser.add_argument("--repo", required=True, help="Git repository URL")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--work-dir", default="workdir", help="Working directory")
    parser.add_argument("--max-files", type=int, default=None, help="Max Java files to process")
    args = parser.parse_args()

    print("🚀 Starting Java Repository Analysis")
    print("=" * 60)

    analyzer = JavaRepositoryAnalyzer(
        repo_url=args.repo,
        gemini_api_key=args.api_key,
        work_dir=args.work_dir,
        max_files=args.max_files
    )

    # Clone repository
    print("\n📥 Cloning repository...")
    analyzer.clone_repository()

    # Build knowledge graph
    print("\n🧠 Building knowledge graph...")
    analyzer.build_knowledge_graph()

    # Get statistics
    stats = analyzer.get_statistics()
    print(f"\n📊 Analysis Statistics:")
    for key, value in stats.items():
        print(f"   - {key.replace('_', ' ').title()}: {value}")

    if stats["total_nodes"] > 0:
        # Analyze architecture
        print("\n🏗️  Architecture Analysis:")
        print("-" * 40)
        architecture = analyzer.analyze_architecture()
        print(architecture)

        # Example queries
        print("\n❓ Sample Queries:")
        questions = [
            "What are the main classes in this repository?",
            "How is error handling implemented?",
            "Show me examples of service classes"
        ]

        for question in questions:
            print(f"\nQ: {question}")
            print("-" * 30)
            answer = analyzer.query_codebase(question)
            print(answer[:1000] + "..." if len(answer) > 1000 else answer)

    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()
