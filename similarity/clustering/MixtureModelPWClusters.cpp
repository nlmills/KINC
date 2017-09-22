#include "MixtureModelPWClusters.h"

/**
 * Constructor for MixMod class.
 *
 * @param EMatrix *ematrix;
 */
MixtureModelPWClusters::MixtureModelPWClusters(PairWiseSet *pwset, int min_obs,
    char ** method, int num_methods) {

  // Initialize some values.
  this->pwset = pwset;
  this->min_obs = min_obs;
  this->method = method;
  this->num_methods = num_methods;

  // Create the PairWiseClusterList object
  this->pwcl = new PairWiseClusterList(pwset);

  // Make sure we have the correct number of observations before preparing
  // for the comparision.
  if (this->pwset->n_clean < this->min_obs) {
    return;
  }

  // Create the Gaussian Data object and set the dataDescription object.
  this->data = ML::Matrix(2, pwset->n_clean);

  for ( int i = 0; i < pwset->n_clean; i++ ) {
    this->data.elem(0, i) = pwset->x_clean[i];
    this->data.elem(1, i) = pwset->y_clean[i];
  }
}

/**
 * Destructor for the MixMod class.
 */
MixtureModelPWClusters::~MixtureModelPWClusters() {
}

/**
 * Executes mixture model clustering.
 */
void MixtureModelPWClusters::run(char * criterion, int max_clusters) {
  // Make sure we have the correct number of observations before performing
  // the comparision.
  if (this->pwset->n_clean < this->min_obs) {
    return;
  }

  // construct clustering layers for 1..max_clusters
  std::vector<ML::ClusteringLayer *> clus_layers;
  for ( int i = 1; i <= max_clusters; i++ ) {
    clus_layers.push_back(new ML::GMMLayer(i));
  }

  // construct criterion layer
  ML::CriterionLayer *crit_layer;

  if (strcmp(criterion, "BIC") == 0) {
    crit_layer = new ML::BICLayer();
  }
  else if (strcmp(criterion, "ICL") == 0) {
    crit_layer = new ML::ICLLayer();
  }

  // run clustering
  ML::ClusteringModel model(clus_layers, crit_layer);
  model.predict(this->data);

  if ( model.best_layer() == nullptr ) {
    return;
  }

  // get clustering output
  this->labels = model.best_layer()->output();
  int num_clusters = model.best_layer()->num_clusters();

  // temporary code to make labels 1-based
  for ( size_t i = 0; i < this->labels.size(); i++ ) {
    this->labels[i]++;
  }

  // Iterate through the clusters in the set that was selected.  We are
  // done iterating through the clusters when we run out of labels in the
  // labels array.
  int cluster_num = 1;
  int cluster_samples[this->pwset->n_clean];
  bool done = false;
  while (!done) {
    // As a default set done to be true. If we find a label for the current
    // cluster number then it will get set to false and we can deal with the
    // cluster.
    done = true;

    // Prepare arrays for outlier detection and removal.
    float cx[this->pwset->n_clean];
    float cy[this->pwset->n_clean];
    for (int j = 0; j < this->pwset->n_clean; j++) {
      cx[j] = 0;
      cy[j] = 0;
    }

    // Build the samples array of 0's and 1's to indicate which
    // samples are in the cluster and which are not, also populate the
    // cx and cy arrays for outlier detection.
    int l = 0;
    for (int i = 0; i < this->pwset->n_clean; i++) {
      if (this->labels[i] == cluster_num) {
        cx[l] = this->pwset->x_clean[i];
        cy[l] = this->pwset->y_clean[i];
        l++;
        done = false;
        cluster_samples[i] = 1;
      }
      else {
        cluster_samples[i] = 0;
      }
    }

    // Discover any outliers for clusters with size >= min_obs
    Outliers * outliersCx = NULL;
    Outliers * outliersCy = NULL;
    outliersCx = outliers_iqr(cx, this->pwset->n_clean, 1.5);
    outliersCy = outliers_iqr(cy, this->pwset->n_clean, 1.5);

    // Remove any outliers
    for (int i = 0; i < this->pwset->n_clean; i++) {
      for (int j = 0; j < outliersCx->n; j++) {
        if (cluster_samples[i] == 1 && this->pwset->x_clean[i] == outliersCx->outliers[j]) {
          cluster_samples[i] = 8;
        }
      }
      for (int j = 0; j < outliersCy->n; j++) {
        if (cluster_samples[i] == 1 && this->pwset->y_clean[i] == outliersCy->outliers[j]) {
          cluster_samples[i] = 8;
        }
      }
    }
    if (outliersCx) {
      free(outliersCx->outliers);
      free(outliersCx);
    }
    if (outliersCy) {
      free(outliersCy->outliers);
      free(outliersCy);
    }

    // If we found samples with the current cluster_num then create a
    // cluster and add it to the list.
    if (!done) {
      PairWiseCluster * cluster = new PairWiseCluster(this->pwset, this->method, this->num_methods);
      cluster->setClusterSamples(cluster_samples, true);
      cluster->doSimilarity(this->min_obs);
      this->pwcl->addCluster(cluster);
    }
    cluster_num++;
  }

  // cleanup
  while ( !clus_layers.empty() ) {
      delete clus_layers.back();
      clus_layers.pop_back();
  }
  delete crit_layer;
}
