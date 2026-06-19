import json
import os
import base64

NOTEBOOKS = [
    "06_Explainability.ipynb",
    "07_Advanced_Evaluation.ipynb"
]

import paths_config
paths_config.ensure_dirs()
PLOTS_DIR = str(paths_config.OUTPUTS_PLOTS)

def ensure_plots_dir():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"📁 Created directory: {PLOTS_DIR}")

def extract_images_from_outputs(notebook_name, cells):
    """Extracts existing base64 images from cell outputs."""
    count = 0
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        
        outputs = cell.get("outputs", [])
        for j, output in enumerate(outputs):
            data = output.get("data", {})
            if "image/png" in data:
                # Found an image
                img_data = data["image/png"]
                img_bytes = base64.b64decode(img_data)
                
                filename = f"{notebook_name}_cell_{i}_output_{j}.png"
                filepath = os.path.join(PLOTS_DIR, filename)
                
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
                
                print(f"  🖼️  Extracted existing plot: {filepath}")
                count += 1
    return count

def inject_savefig(notebook_name, cells):
    """Injects plt.savefig() into code cells that produce plots."""
    modified_count = 0
    
    # Heuristics for plotting cells
    plot_indicators = ["plt.plot", "sns.", "plot(", "plt.show", "confusion_matrix", "roc_curve"]
    
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
            
        source_lines = cell.get("source", [])
        source_text = "".join(source_lines)
        
        # Check if cell likely plots something
        if any(indicator in source_text for indicator in plot_indicators):
            # Check if already has savefig to avoid duplicates
            if "plt.savefig" in source_text:
                continue
                
            print(f"  ✏️  Injecting savefig into Cell {i}...")
            
            # Define save command
            clean_name = notebook_name.replace(".ipynb", "")
            save_cmd = f"plt.savefig('{PLOTS_DIR}/{clean_name}_cell_{i}.png', bbox_inches='tight')\n"
            
            new_source = []
            injected = False
            
            # Strategy: Inject before plt.show() if present, else append at end
            if "plt.show()" in source_text:
                for line in source_lines:
                    if "plt.show()" in line:
                        # Insert before the line with plt.show()
                        # Handle indentation? Usually plt.show() is at base level or same indent.
                        # Simple approach: Insert save_cmd before line
                        # Try to match indentation
                        indent = line[:len(line) - len(line.lstrip())]
                        new_source.append(f"{indent}{save_cmd}")
                        new_source.append(line)
                        injected = True
                    else:
                        new_source.append(line)
            else:
                # Append to end
                new_source = source_lines
                # Ensure newline before appending if needed
                if new_source and not new_source[-1].endswith("\n"):
                    new_source[-1] = new_source[-1] + "\n"
                new_source.append(f"\n{save_cmd}")
                injected = True
            
            if injected:
                cell["source"] = new_source
                modified_count += 1
                
    return modified_count

def process_notebook(filename):
    print(f"📘 Processing {filename}...")
    
    if not os.path.exists(filename):
        print(f"  ❌ File not found: {filename}")
        return

    try:
        with open(filename, "r", encoding="utf-8") as f:
            nb = json.load(f)
        
        cells = nb.get("cells", [])
        
        # 1. Extract existing
        extracted = extract_images_from_outputs(filename.replace(".ipynb", ""), cells)
        print(f"  ✅ Extracted {extracted} images.")
        
        # 2. Inject code
        injected = inject_savefig(filename, cells)
        print(f"  ✅ Injected save commands into {injected} cells.")
        
        # Save back
        if injected > 0:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1) # indent=1 to keep file size reasonable but readable
            print(f"  💾 Saved updated notebook: {filename}")
        else:
            print("  Tb No code changes needed.")
            
    except Exception as e:
        print(f"  ❌ Error processing {filename}: {e}")

def main():
    ensure_plots_dir()
    for nb in NOTEBOOKS:
        process_notebook(nb)
    print("\n✨ Done! Plots extracted and notebooks updated.")

if __name__ == "__main__":
    main()
