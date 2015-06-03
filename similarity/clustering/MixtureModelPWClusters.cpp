#include "MixtureModelPWClusters.h"

/**
 * Constructor for MixMod class.
 *
 * @param EMatrix *ematrix;
 */
MixtureModelPWClusters::MixtureModelPWClusters(PairWiseSet *pwset, int min_obs, char * method) {

  // Initialize some values.
  this->pwset = pwset;
  this->min_obs = min_obs;
  this->method = method;
  labels = NULL;
  pwcl = NULL;
  data = NULL;

  // Create the PairWiseClusterList object
  pwcl = new PairWiseClusterList(pwset);

  // Make sure we have the correct number of observations before preparing
  // for the comparision.
  if (this->pwset->n_clean < this->min_obs) {
    return;
  }

  // Create the Gaussian Data object and set the dataDescription object.
  this->data = (double **) malloc(sizeof(double *) * pwset->n_clean);
  for (int i = 0; i < pwset->n_clean ; i++) {
    this->data[i] = (double *) malloc(sizeof(double) * 2);
    this->data[i][0] = pwset->x_clean[i];
    this->data[i][1] = pwset->y_clean[i];
  }

}

/**
 * Destructor for the MixMod class.
 */
MixtureModelPWClusters::~MixtureModelPWClusters() {
  // Free the data array.
  if (data) {
    for (int i = 0; i < pwset->n_clean; i++) {
      free(data[i]);
    }
    free(data);
  }
  if (labels) {
    free(labels);
  }
  delete pwcl;
}

/**
 * Executes mixture model clustering.
 */
void MixtureModelPWClusters::run(char * criterion, int max_clusters) {
  int64_t lowest_cluster_num = 9;
  int found = 0;

  // Make sure we have the correct number of observations before performing
  // the comparision.
  if (this->pwset->n_clean < this->min_obs) {
    return;
  }

  // The data we are using is qualitative, so set the data type.
  XEM::GaussianData * gdata = new XEM::GaussianData(pwset->n_clean, 2, data);
  XEM::DataDescription dataDescription(gdata);

  // Set the number of clusters to be tested to 9.
  vector<int64_t> nbCluster;
  for (int i = 1; i <= max_clusters; i++) {
    nbCluster.push_back(i);
  }

  // Create the ClusteringInput object.
  XEM::ClusteringInput cInput(nbCluster, dataDescription);

  // Set the criterion to ICL
  cInput.removeCriterion(0);
  if (strcmp(criterion, "BIC") == 0) {
    cInput.addCriterion(XEM::BIC);
  }
  else if (strcmp(criterion, "ICL") == 0) {
    cInput.addCriterion(XEM::ICL);
  }
  else if (strcmp(criterion, "NEC") == 0) {
    cInput.addCriterion(XEM::NEC);
  }
  else if (strcmp(criterion, "CV") == 0) {
    cInput.addCriterion(XEM::CV);
  }
  else if (strcmp(criterion, "DCV") == 0) {
    cInput.addCriterion(XEM::DCV);
  }

  // Finalize input: run a series of sanity checks on it
  cInput.finalize();

  // Run XEM::ClusteringMain 5 times to find the iteration with the
  // lowest number of clusters.  We do this because the MixModLib can
  // detect different numbers of clusters depending on the random starting
  // point in the algorithm.
  for (int i = 0; i < 1; i++) {

    // Find clusters.
    XEM::ClusteringMain cMain(&cInput);
    cMain.run();

    // Create a new XEM::ClusteringOutput object
    XEM::ClusteringOutput * cOutput = cMain.getOutput();

    // Order the clusters using BIC.
    if (strcmp(criterion, "BIC") == 0) {
      cOutput->sort(XEM::BIC);
    }
    else if (strcmp(criterion, "ICL") == 0) {
      cOutput->sort(XEM::ICL);
    }
    else if (strcmp(criterion, "NEC") == 0) {
      cOutput->sort(XEM::NEC);
    }
    else if (strcmp(criterion, "CV") == 0) {
      cOutput->sort(XEM::CV);
    }
    else if (strcmp(criterion, "DCV") == 0) {
      cOutput->sort(XEM::DCV);
    }

    if (cOutput->atLeastOneEstimationNoError()) {
      found = 1;

      // Get the best XEM::ClusteringModelOutput
      XEM::ClusteringModelOutput* cMOutput = cOutput->getClusteringModelOutput().front();
      XEM::LabelDescription * ldescription = cMOutput->getLabelDescription();
      XEM::Label * label = ldescription->getLabel();
      int64_t * cLabels = label->getTabLabel();

      // Find the number of clusters
      int64_t num_clusters = 0;
      for (int i = 0; i < this->pwset->n_clean; i++) {
        if (cLabels[i] > num_clusters) {
          num_clusters = cLabels[i];
        }
      }

      // If this is the fewest number of clusters we've seen then copy
      // the labels over.
      if (num_clusters < lowest_cluster_num) {
        lowest_cluster_num = num_clusters;
        if (this->labels == NULL) {
          this->labels = (int64_t *) malloc(sizeof(int64_t) * this->pwset->n_clean);
        }
        for (int i = 0; i < this->pwset->n_clean; i++) {
          this->labels[i] = cLabels[i];
        }
      }
      delete cLabels;
    }
  }
  delete gdata;
  if (found) {
//    printf("Lowest: %d\n", lowest_cluster_num);
    // Create the cluster objects using the iteration with the fewest clusters.
    int cluster_num = 1;
    int cluster_samples[this->pwset->n_clean];
    bool done = false;
    while (!done) {
      done = true;
      for (int i = 0; i < this->pwset->n_clean; i++) {
        if (this->labels[i] == cluster_num) {
          done = false;
          cluster_samples[i] = 1;
        }
        else {
          cluster_samples[i] = 0;
        }
      }
      // if we found samples with the current cluster_num then create a
      // cluster and add it to the list
      if (!done) {
        PairWiseCluster * cluster = new PairWiseCluster(this->pwset);
        cluster->setClusterSamples(cluster_samples, true);
        // Calculate the Spearman correlation for this cluster.
        cluster->doSimilarity(this->method, this->min_obs);
//        cluster->printCluster();
        this->pwcl->addCluster(cluster);
      }
      cluster_num++;
    }
  }
}