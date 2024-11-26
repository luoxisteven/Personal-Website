import React, { useEffect, useState, useContext } from "react";
import axios from "axios";
import Markdown from "mui-markdown";
import { Box, Typography, useTheme } from "@mui/material";
import { SettingsContext } from "../context/SettingsContext";

const MarkdownViewer = ({ filePath }) => {
  const [markdownContent, setMarkdownContent] = useState("");
  const { themeMode } = useContext(SettingsContext); // 获取主题模式
  const theme = useTheme(); // Material-UI 主题

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
    <Box sx={{ p: 3, margin: "0 auto" }}>
      <Box
        sx={{
          borderRadius: "8px",
          p: 2,
        //   border: `1px solid ${
        //     themeMode === "light"
        //       ? theme.palette.grey[300] // 浅色模式下，边框颜色浅灰色
        //       : theme.palette.grey[800] // 深色模式下，边框颜色稍深灰色
        //   }`,
        //   backgroundColor: themeMode === "light" ? theme.palette.grey[50] : theme.palette.grey[900], // 背景颜色
        }}
      >
        {markdownContent ? (
          <Markdown
            overrides={{
              h1: {
                component: Typography,
                props: { variant: "h5", gutterBottom: true, sx: { fontWeight: 600 } },
              },
              h2: {
                component: Typography,
                props: { variant: "h6", gutterBottom: true, sx: { fontWeight: 600 } },
              },
              h3: {
                component: Typography,
                props: { variant: "subtitle1", gutterBottom: true, sx: { fontWeight: 600 } },
              },
              p: {
                component: Typography,
                props: { variant: "body", gutterBottom: true, sx: { fontSize: "0.9rem" } },
              },
              li: {
                component: Typography,
                props: { component: "li", variant: "body", gutterBottom: true, sx: { fontSize: "0.9rem" } },
              },
            }}
          >
            {markdownContent}
          </Markdown>
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
