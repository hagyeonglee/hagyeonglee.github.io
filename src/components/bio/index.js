import React from 'react';
import ReactRotatingText from 'react-rotating-text';
import IconButtonBar from '../icon-button-bar';
import Image from '../image';
import './style.scss';

function Bio({ author, language = 'ko' }) {
  if (!author) return null;
  const { bio, social, name } = author;
  return (
    <div className="bio">
      {language === 'ko' ? (
        <div className="introduction korean">
          <p className="title">
            안녕하세요 :)
            <br />
            <ReactRotatingText items={bio.description} />
            <br />
            {bio.role} <strong>{name}</strong>입니다.
            <br />
          </p>
          <div className="social-links">
            <IconButtonBar links={social} />
          </div>
        </div>
      ) : (
        <div className="introduction english">
          <p className="title">
            Hello 🙌
            <br />
            I am
            <br />
            <strong>Hagyeong</strong>
            <br />
          </p>
          <p className="description">
            <br />
            <ReactRotatingText items={bio.description} />
            <br />
            {bio.role}
            <br />
          </p>
          <div className="social-links">
            <IconButtonBar links={social} />
          </div>
        </div>
      )}
      <div className="thumbnail-wrapper">

        <Image src={bio.thumbnail} alt="thumbnail" />

      </div>

    </div>

  );
}

export default Bio;
