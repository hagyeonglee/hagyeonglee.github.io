import React from 'react';
import './style.scss';

function PageFooter({ author, githubUrl }) {
  return (
    <footer className="page-footer-wrapper">
      <p className="page-footer">
        © {new Date().getFullYear()}
        &nbsp;
        <a href={githubUrl}>{author}</a>
        &nbsp;powered by
        {/* https://zoomkod.ing/gatsby-github-blog/ */}
        <a href="https://github.com/hagyeonglee/hagyeonglee.github.io">
          &nbsp;hagyeonglee
        </a>
      </p>
    </footer>
  );
}

export default PageFooter;
