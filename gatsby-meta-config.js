module.exports = {
  title: `Gyoonglog`,
  description: `배우고 기록하고 성장하자!`,
  language: `ko`, // `ko`, `en` => currently support versions for Korean and English
  siteUrl: `https://hagyeong-lee.netlify.app/`,
  ogImage: `/og-image.png`, // Path to your in the 'static' folder
  comments: {
    utterances: {
      repo: ``, // `zoomkoding/zoomkoding-gatsby-blog`,
    },
  },
  ga: '0', // Google Analytics Tracking ID
  author: {
    name: `Hagyeonglee`,
    bio: {
      role: `AI Researcher`,
      description: ['사람에 가치를 두는', '능동적으로 일하는', '이로운 것을 만드는'],
      thumbnail: 'blog_profile.png', // Path to the image in the 'asset' folder
    },
    social: {
      //instagram: `https://www.instagram.com/leee_eeehg/`,
      github: `https://github.com/hagyeonglee`, // `https://github.com/hagyeonglee`,
      linkedIn: `https://www.linkedin.com/in/hagyeong-lee-1b342520b/`, // `https://www.linkedin.com/in/hagyeong-lee-1b342520b/`,
      email: `lhky0708@gmail.com`, // `lhky0708@gmail.com`,
      // CV: ``,
      //koreanResume: `/CV_hagyeonglee.pdf`,
      CurriculumVitae: `/CV_hagyeonglee.pdf`,

    },
  },

  // metadata for About Page
  about: {
    timestamps: [
      // =====       [Timestamp Sample and Structure]      =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!) =====
      {
        date: '',
        activity: '',
        links: {
          github: '',
          post: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
      {
        date: '2022.02 ~',
        activity: '배우고 기록하는 블로그 시작',
        links: {
          post: '',
          github: 'https://github.com/hagyeonglee/hagyeonglee-gatsby-blog',
          demo: 'https://hagyeong-lee.netlify.app/',
        },
      },
    ],

    projects: [
      // =====        [Project Sample and Structure]        =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!)  =====
      {
        title: '',
        description: '',
        techStack: ['', ''],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
      {
        title: '',
        description:
          '',
        techStack: ['', ''],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          demo: '',
        },
      },
    ],
  },
};
