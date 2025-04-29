import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Effortless Integration',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        RAG Chatbot is designed for effortless setup and integration. Simply connect your external knowledge sources, and the chatbot will start delivering accurate, context-aware responses to user queries-no complex configuration required
      </>
    ),
  },
  {
    title: 'Contextual, Accurate Answers',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
       Let you concentrate on curating and updating your knowledge base, while RAG Chatbot automatically retrieves the most relevant information and generates precise answers. This ensures users always receive up-to-date, reliable responses based on your data
      </>
    ),
  },
  {
    title: 'Driven by RAG Technology',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        This reflect the unique strengths of RAG chatbots-namely, their ability to combine generative AI with dynamic, external knowledge retrieval for more accurate and reliable user interactions.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
