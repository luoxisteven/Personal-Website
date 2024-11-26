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
        console.log(filePath);
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
      <Typography variant="h4" gutterBottom>
        Markdown Viewer
      </Typography>
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
                <Typography variant="h4" gutterBottom {...props} />
              ),
              h2: ({ node, ...props }) => (
                <Typography variant="h5" gutterBottom {...props} />
              ),
              h3: ({ node, ...props }) => (
                <Typography variant="h6" gutterBottom {...props} />
              ),
              p: ({ node, ...props }) => (
                <Typography variant="body1" gutterBottom {...props} />
              ),
              li: ({ node, ...props }) => (
                <Typography
                  component="li"
                  variant="body1"
                  gutterBottom
                  {...props}
                />
              ),
            }}
          />
        ) : (
          <Typography variant="body1" color="textSecondary">
            正在加载内容...
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default MarkdownViewer;
