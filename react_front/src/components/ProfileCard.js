import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Grid,
  IconButton,
  Box,
} from "@mui/material";
import EmailIcon from "@mui/icons-material/Email";
import InstagramIcon from "@mui/icons-material/Instagram";
import FlickrIcon from "@mui/icons-material/PhotoLibrary";
import SchoolIcon from "@mui/icons-material/School";
import LinkIcon from "@mui/icons-material/Link";

const ProfileCard = () => {
  // 使用 useState 管理头像路径
  const [avatar, setAvatar] = useState(require("../assets/img/steven.JPG"));

  // 切换图片的处理函数
  const handleAvatarClick = () => {
    setAvatar((prevAvatar) =>
      prevAvatar === require("../assets/img/steven.JPG")
        ? require("../assets/img/steven_cartoon_2.JPG")
        : require("../assets/img/steven.JPG")
    );
  };

  return (
    <Card
      sx={{
        maxWidth: 345,
        padding: 2,
        textAlign: "center",
        boxShadow: "none !important", // 强制移除阴影
        background: "transparent !important", // 添加 background 属性
      }}
    >
      {/* 头像 */}
      <CardMedia
        component="img"
        alt="Steven Luo"
        src={avatar}
        onClick={handleAvatarClick} // 添加点击事件处理函数
        sx={{
          width: 200,
          height: 200,
          margin: "0 auto",
          borderRadius: "50%", // 圆形效果
          cursor: "pointer", // 添加鼠标指针样式
        }}
      />
      {/* 内容 */}
      <CardContent>
        <Typography variant="h6" component="div">
          Xi Luo
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Junior Machine Learning Engineer
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          @addaxis.ai
        </Typography>
      </CardContent>

      {/* 按钮区域 */}
      <Box>
        <Grid container justifyContent="center" spacing={2}>
          <Grid item>
            <IconButton href="#" color="primary" aria-label="email">
              <EmailIcon />
            </IconButton>
          </Grid>
          <Grid item>
            <IconButton href="#" color="secondary" aria-label="instagram">
              <InstagramIcon />
            </IconButton>
          </Grid>
          <Grid item>
            <IconButton href="#" color="default" aria-label="flickr">
              <FlickrIcon />
            </IconButton>
          </Grid>
          <Grid item>
            <IconButton href="#" color="info" aria-label="google scholar">
              <SchoolIcon />
            </IconButton>
          </Grid>
          <Grid item>
            <IconButton href="#" color="default" aria-label="orcid">
              <LinkIcon />
            </IconButton>
          </Grid>
        </Grid>
      </Box>
    </Card>
  );
};

export default ProfileCard;