import React, { useEffect, useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Box, Typography } from "@mui/material";
import "katex/dist/katex.min.css"; 

const MarkdownViewer = ({ filePath }) => {
  const [markdownContent, setMarkdownContent] = useState("");

  useEffect(() => {
    const loadMarkdown = async () => {
      try {
        const response = await axios.get(filePath);
        setMarkdownContent(response.data);
      } catch (error) {
        setMarkdownContent("无法加载指定的 Markdown 文件。");
        console.error(error);
      }
    };

    if (filePath) {
      loadMarkdown();
    }
  }, [filePath]);

  return (
    <Box sx={{ p: 4, maxWidth: "800px", margin: "0 auto" }}>
      <Box
        sx={{
          border: "1px solid #ccc",
          borderRadius: "8px",
          p: 2,
        }}
      >
        {markdownContent ? (
          <ReactMarkdown
            children={markdownContent}
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              h1: ({ node, ...props }) => (
                <Typography
                  variant="h5"
                  gutterBottom
                  sx={{ fontSize: "1.5rem", fontWeight: "bold" }}
                  {...props}
                />
              ),
              h2: ({ node, ...props }) => (
                <Typography
                  variant="h6"
                  gutterBottom
                  sx={{ fontSize: "1.25rem", fontWeight: 600 }}
                  {...props}
                />
              ),
              h3: ({ node, ...props }) => (
                <Typography
                  variant="subtitle1"
                  gutterBottom
                  sx={{ fontSize: "1.1rem", fontWeight: 600 }}
                  {...props}
                />
              ),
              p: ({ node, ...props }) => (
                <Typography
                  variant="body2"
                  gutterBottom
                  sx={{ fontSize: "0.95rem" }}
                  {...props}
                />
              ),
              li: ({ node, ...props }) => (
                <Typography
                  component="li"
                  variant="body2"
                  gutterBottom
                  sx={{ fontSize: "0.95rem" }}
                  {...props}
                />
              ),
            }}
          />
        ) : (
          <Typography variant="body2" color="textSecondary">
            正在加载内容...
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default MarkdownViewer;
