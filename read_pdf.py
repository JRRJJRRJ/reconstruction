#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时脚本：读取PDF论文内容"""

import sys
import os

def read_pdf(pdf_path):
    """尝试使用多种方法读取PDF"""
    text_content = []
    
    # 方法1: 尝试使用PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_content.append(f"\n=== Page {page_num + 1} ===\n{text}")
        return "\n".join(text_content)
    except ImportError:
        pass
    except Exception as e:
        print(f"PyPDF2 failed: {e}", file=sys.stderr)
    
    # 方法2: 尝试使用pypdf
    try:
        import pypdf
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_content.append(f"\n=== Page {page_num + 1} ===\n{text}")
        return "\n".join(text_content)
    except ImportError:
        pass
    except Exception as e:
        print(f"pypdf failed: {e}", file=sys.stderr)
    
    # 方法3: 尝试使用pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_content.append(f"\n=== Page {page_num + 1} ===\n{text}")
        return "\n".join(text_content)
    except ImportError:
        pass
    except Exception as e:
        print(f"pdfplumber failed: {e}", file=sys.stderr)
    
    return None

if __name__ == "__main__":
    pdf_path = "基于社区划分和关键节点的高阶网络重构 (3).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    
    content = read_pdf(pdf_path)
    
    if content:
        # 输出到文件
        output_file = "论文内容提取.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"PDF content extracted to: {output_file}")
        print(f"\nFirst 2000 characters:\n{content[:2000]}")
    else:
        print("Error: Could not read PDF. Please install one of: PyPDF2, pypdf, or pdfplumber", file=sys.stderr)
        print("Or provide Word version of the paper.", file=sys.stderr)
        sys.exit(1)

