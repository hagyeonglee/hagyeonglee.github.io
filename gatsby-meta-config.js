module.exports = {
  title: `Hagyeong-`,
  description: `배우고 기록하고 성장하자!`,
  language: `en`, // `ko`, `en` => currently support versions for Korean and English
  siteUrl: `https://hagyeonglee.github.io/`,
  ogImage: `/og-image.png`, // Path to your in the 'static' folder
  comments: {
    utterances: {
      repo: `hagyeonglee/comments`, // `zoomkoding/zoomkoding-gatsby-blog`,
    },
  },
  ga: 'G-Z1P475ZSBN', // Google Analytics Tracking ID
  author: {
    name: `Hagyeong Lee`,
    bio: {
      role: `AI Traveler`,
      description: ['몰랑한 호기심을 가진', '말랑하게 사고하는', '꼼꼼하게 성장하는'],
      thumbnail: 'github_profile.png', // Path to the image in the 'asset' folder
    },
    social: {
      instagram: `https://www.instagram.com/leee_eeehg/`,
      github: `https://github.com/hagyeonglee`, // `https://github.com/hagyeonglee`,
      linkedIn: `https://www.linkedin.com/in/hagyeong-lee-1b342520b/`, // `https://www.linkedin.com/in/hagyeong-lee-1b342520b/`,
      email: `hagyeonglee@postech.ac.kr`, // `lhky0708@gmail.com`,
      CurriculumVitae: `/CV_hagyeonglee.pdf`,
      PersonalBlog: `https://blog.naver.com/hagyng`,
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
          instagram: '',
          file: '',

        },
      },
      // ========================================================
      // ========================================================
      //
      {
        date: '2022. 09 ~ ',
        activity: 'M.S. Candidate Student @POSTECH EffL Lab.',
        links: {
          post: 'https://effl.postech.ac.kr/',
          github: '',
          demo: '',
          file: '',
        },
      },
      {
        date: '2022. 01 ~ 02',
        activity: 'POSTECH E.E. Winter Research Intern',
        links: {
          post: '',
          github: '',
          demo: '',
          file: '',
        },
      },
      {
        date: '2021. 09 ~ 2021. 12',
        activity: '한국과학기술연구원(KIST) 차세대반도체연구소 스핀융합연구단 Research Intern',
        links: {
          post: '',
          github: '',
          demo: '',
          file: '',
        },
      },
      {
        date: '2020. 07 ~ 2021. 08',
        activity: 'GoogleDeveloper Student Club Ewha Lead',
        links: {
          post: '',
          github: 'https://github.com/DSC-Ewha',
          demo: 'https://gdsc.community.dev/ewha-womans-university/',
          file: '',
        },
      },
      {
        date: '2021. 01 ~ 2021. 12',
        activity: 'Ewha Womans University C.S.E Capstone Project Leader ',
        links: {
          post: '',
          github: 'https://github.com/TripleH-EwhaCSE/Mein_Flutter',
          demo: '',
          file: '',
        },
      },
      {
        date: '2020. 09 ~ 2020. 12',
        activity: '연세 세브란스 병원 Center for Clinical Imaging Data Science(CCIDS) Research Intern',
        links: {
          post: '',
          github: '',
          demo: '',
          file: '',
        },
      },
      {
        date: '2020.07 ~ 2020.08',
        activity: 'Ewha GraphicsLAB Research Intern',
        links: {
          post: '',
          github: '',
          demo: '',
          file: 'https://drive.google.com/file/d/10V59AvDk_J3MDY6hd3ym8mXNjp4LN2d6/view?usp=sharing',
        },
      },
      {
        date: '2018.03 ~',
        activity: 'Ewha Womans University',
        links: {
          post: '',
          github: '',
          demo: '',
          file: '/CV_hagyeonglee.pdf',
        },
      },
      {
        date: '2018.03 ~',
        activity: 'Curriculum Vitae :)',
        links: {
          post: '',
          github: '',
          demo: '',
          file: '/CV_hagyeonglee.pdf',
        },
      },
      {
        date: '',
        activity: 'Start !',
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
          instagram: '',
          file: '',
        },
      },
      // ========================================================
      // ========================================================
      {
        title: 'POSTECH E.E. Winter Undergraduate Research Fellowship',
        description:
          'Study aboout Implicit Neural Representations for Image Compression  ',
        techStack: ['ML', 'Compression'],
        thumbnailUrl: '',
        links: {
          post: 'https://uncovered-panda-d64.notion.site/Implicit-Neural-Representations-for-Image-Compression-5a6c3f0f23b247e5ae4b89757333a29c',
          github: 'https://github.com/hagyeonglee/ModelCompression',
          demo: '',
          instagram: '',
          file: '',
        },
      },
      {
        title: 'KIST(한국과학기술연구원) 차세대반도체연구소 Undergraduate Research Program ',
        description:
          'Machine learning that can predict Imbalanced tabular data and obtain important factors as a result',
        techStack: ['ML', 'Tabular Data'],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          demo: '',
          instagram: '',
          file: '',
        },
      },
      {
        title: 'Undergraduate Research Intern @Yonsei Severance CCIDS',
        description: 'Deep learning model for C-shape detection in dental radiographs ',
        techStack: ['ML', 'Medical Image'],
        thumbnailUrl: '',
        links: {
          post: 'https://docs.google.com/document/d/1qSKga29gllONG5PG7l1cn9Ewo5QiADi_/edit?usp=sharing&ouid=105952325702141881372&rtpof=true&sd=true',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: '',
          instagram: '',
          file: '',
        },
      },
      {
        title: '메뉴판 촬영을 통한 AI 한식 검색 어플리케이션;MeIN(Menu Intuition)',
        description: 'Korean food explanation and recommendation application using deep learning, Ewha SW Graduation Project | (presentation 1st , Poster 3rd )',
        techStack: ['Application Develop(Flutter)', 'ML', 'Data pre-processing'],
        thumbnailUrl: '',
        links: {
          post: '',
          github: 'https://github.com/TripleH-EwhaCSE/Mein_Flutter',
          googlePlay: 'https://play.google.com/store/apps/details?id=com.ohjoo.mein&hl=ko&gl=US',
          appStore: '',
          demo: '',
          instagram: '',
          file: '',
        },
      },
      {
        title: 'Challenge Semester Research Program (Ewha 도전학기)',
        description: 'Deep Learning based Signal Pattern Design of Abandoned Dog(유기견) Audio',
        techStack: ['ML', 'Audio processing', 'Start-up'],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: 'https://tumblbug.com/happycalender',
          instagram: '',
          file: '',
        },
      },
      {
        title: 'Undergraduate Research Intern @Ewha Graphics Lab',
        description: 'A Simulation-Based Convergence Content: AI Collaborated with Human - Robot Drawing Model ',
        techStack: ['ML', 'Generative Model'],
        thumbnailUrl: '',
        links: {
          post: 'https://drive.google.com/file/d/10V59AvDk_J3MDY6hd3ym8mXNjp4LN2d6/view?usp=sharing',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: '',
          instagram: '',
          file: '',
        },
      },
      {
        title: 'Ewha Start-up Internship : Toonsquare(Tooning)',
        description: 'Text to Toon Emotion using NLP to return emotions from text by classifying them into seven types labels',
        techStack: ['ML', 'NLP', 'FrontEnd(Typescript)'],
        thumbnailUrl: '',
        links: {
          post: 'https://tooning.io/template-list/home',
          github: 'https://github.com/hagyeonglee/toonsquare_ai_kobert',
          googlePlay: 'https://play.google.com/store/apps/details?id=com.tooning.app',
          appStore: '',
          demo: 'http://stage.toonsquare.co/ai/emotion/kor?tester',
          instagram: '',
          file: '',
        },
      },
      {
        title: 'Google Developer Student Club Ewha Lead',
        description: 'Planning and operating various development programs(DevFest, Hackathon, Pep talk) and events conducted by Google Korea.',
        techStack: ['Communication Skill', 'ML', 'FrontEnd(React)'],
        thumbnailUrl: '',
        links: {
          post: 'https://drive.google.com/file/d/121rsKHUGWERt3ZTso8UitFcn3G-uPB1z/view?usp=sharing',
          github: 'https://github.com/hagyeonglee/FrontEnd-planetEarth-',
          github: 'https://github.com/DSC-Ewha/ReactGlobe',
          googlePlay: '',
          appStore: '',
          demo: '',
          instagram: '',
          file: 'https://drive.google.com/file/d/1B4CqQ58SEfAqY_ZfD_bpsuj09H64ILg3/view?usp=sharing',
        },
      },
      {
        title: 'NAEK YEHS Ewha School Representitive',
        description: 'Young Engineers Honor Society (YEHS)는 한국공학한림원 산하의 전국 공대생 네트워크 구축을 위해 학술적으로 교류하는 단체입니다',
        techStack: ['Communication Skill', 'ML'],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: '',
          instagram: '',
          file: '',
        },
      },
      {
        title: 'Ewha Social Start-up & Venture program ; RockleProj. , Rodang Proj.',
        description: 'Content & Communication Application offered to oppertunity for Active Senior(Age 50+ generation) then, reduce depression and role deprivation ',
        techStack: ['Start-up'],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: 'https://youtu.be/H2ozH0vXRQE',
          instagram: '',
          file: 'https://drive.google.com/file/d/1z7OKDq8JlsE9DzyC0he8t4_ZVNfHN_aH/view?usp=sharing',
        },
      },

      // {
      //   title: '',
      //   description: '',
      //   techStack: ['', ''],
      //   thumbnailUrl: '',
      //   links: {
      //     post: '',
      //     github: '',
      //     googlePlay: '',
      //     appStore: '',
      //     demo: '',
      //     instagram: '',
      //     file: '',
      //   },
      // },
      // {
      //   title: '',
      //   description: '',
      //   techStack: ['', ''],
      //   thumbnailUrl: '',
      //   links: {
      //     post: '',
      //     github: '',
      //     googlePlay: '',
      //     appStore: '',
      //     demo: '',
      //     instagram: '',
      //     file: '',
      //   },
      // },
    ],
  },
};
