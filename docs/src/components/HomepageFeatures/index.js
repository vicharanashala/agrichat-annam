import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Seamless Farm Query Integration',
    Svg: require('@site/static/img/farmer_chat_integration.svg').default,
    description: (
      <>
        AgriChat connects directly with agricultural knowledge bases like Kisan Call Centre logs and local farming data. Farmers can ask natural-language questions and receive expert-backed responses instantly—no technical setup required.
      </>
    ),
  },
  {
    title: 'Accurate, Crop-Specific Guidance',
    Svg: require('@site/static/img/crop_insight.svg').default,
    description: (
      <>
       Whether it's pest control for rice or fertilization tips for sugarcane, AgriChat uses retrieval-augmented generation (RAG) to provide answers that are timely, region-aware, and contextually relevant to each farmer's needs.
      </>
    ),
  },
  {
    title: 'Driven by RAG Technology',
    Svg: require('@site/static/img/rag_engine.svg').default,
    description: (
      <>
        By combining large language models with a local vector database of real farmer–expert interactions, AgriChat ensures scalable, privacy-friendly advice grounded in verified agricultural practices and localized crop knowledge.
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
