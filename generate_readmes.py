import os
import ast

folders = ['processing', 'ui', 'visualization', 'tests']

def get_docstring(node):
    doc = ast.get_docstring(node)
    if doc:
        lines = doc.strip().split('\n')
        res = []
        for line in lines:
            if line.strip() == '':
                break
            res.append(line.strip())
        return ' '.join(res)
    return "No description provided."

def get_args(node):
    args = [a.arg for a in node.args.args]
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return ", ".join(args)

def parse_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    module_doc = get_docstring(tree)
    if module_doc == "No description provided.":
        module_doc = ""
    else:
        module_doc = f"_{module_doc}_\n\n"

    items = []
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith('_'): 
                doc = get_docstring(node)
                args_str = get_args(node)
                items.append(f"- **`{node.name}({args_str})`**\n  - {doc}")
        elif isinstance(node, ast.ClassDef):
            doc = get_docstring(node)
            class_info = [f"### `class {node.name}`\n{doc}\n"]
            class_info.append("**Methods:**")
            has_methods = False
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef):
                    if subnode.name == '__init__' or not subnode.name.startswith('_'):
                        meth_doc = get_docstring(subnode)
                        args_str = get_args(subnode)
                        class_info.append(f"- **`{subnode.name}({args_str})`**: {meth_doc}")
                        has_methods = True
            if not has_methods:
                class_info.append("- _No public methods_")
            items.append('\n'.join(class_info))
            
    return module_doc, items

def main():
    for folder in folders:
        if not os.path.exists(folder):
            continue
            
        readme_path = os.path.join(folder, 'README.md')
        
        md_content = f"# {folder.capitalize()} Module\n\n"
        md_content += f"This directory contains detailed information about the functions and classes in the `{folder}` module.\n\n"
        
        files = [f for f in os.listdir(folder) if f.endswith('.py') and f != '__init__.py']
        files.sort()
        
        for file in files:
            filepath = os.path.join(folder, file)
            parsed = parse_file(filepath)
            if parsed:
                module_doc, items = parsed
                md_content += f"## `{file}`\n\n"
                if module_doc:
                    md_content += module_doc
                
                if items:
                    for item in items:
                        md_content += f"{item}\n\n"
                else:
                    md_content += "_No public classes or functions found._\n\n"
                    
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Updated {readme_path}")

if __name__ == '__main__':
    main()