#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

//A helper function to calculate w*g_k in the numerator of equation 6
//Raftery et al (2005)
NumericMatrix wg_k(NumericMatrix mean_k, NumericVector y, NumericVector weights, double sigma) {

  //This function will generate the weights times the pdf
  //This function will generate the numerator in equation (6) of
  //Raftery et al. (2005)

  //The number of time periods
  double T = mean_k.nrow();
  //The number of forecasts in the ensemble
  double K = mean_k.ncol();

  //Initialize temporary doubles
  double temp_y = 0;
  double temp_out = 0;
  double temp_mean_k = 0;
  bool temp_log = false;

  NumericMatrix out(T, K);
  for (int j = 0; j < K; ++j) {
    for (int i = 0; i < T; ++i) {

      temp_y = y[i];
      temp_mean_k = mean_k(i,j);
      temp_out = R::dnorm(temp_y, temp_mean_k, sigma, temp_log);
      temp_out = temp_out * weights[j];
      out(i,j) = temp_out;

    }

  }

  return out;

}

//' The E-M algorithm for Dynamic BMA
//'
//' For the E-M Algorithm for the Dynamic BMA of Rafterey et al.
//' (2005) using the normal distribution. This function will conduct
//' the EM algorithm for maximaziation of the of the
//' Likelihood function as on p. 1159 of Raftery et al. (2005)
//'
//' @title The em algorithm for dynamic bma using the normal distribuiton
//' @param mean_k (NumericMatrix) the bias corrected mean of the forecast
//' for each member of the ensemble (columns) for each
//' time period (rows)
//' @param y (NumericVector) the quantity to be forecast
//' @param squared_error (arma::mat) A matrix with the squared errors for each
//' member of the ensemble (columns) and each time period (rows)
//' @param priors (NumericVector) the prior distribution for the weights (chosen
//' by the user)
//' @param weights (NumericVector) the initial weights to use in the algorithm
//' @param sigma (double) the intial standard deviation for the normal distribution
//' @param maxiter (int) the maximum number of iterations. Defaults to 1000
//' @return A \code{List} with the following elements: \code{weights}, \code{sigma}, and
//' the \code{log_lik}
//' @author Chandler Lutz
//'
// [[Rcpp::export]]
List em(NumericMatrix mean_k, NumericVector y, arma::mat squared_error,
	NumericVector priors, NumericVector weights, double sigma, int maxiter = 1000) {
  /* List = em(mean_k, y, squared_error, weights, sigma, priors)
   *
   * This function will conduct the EM algorithm for maximaziation of the of the
   * Likelihood function as on p. 1159 of Raftery et al. (2005)
   *
   * User-specified inputs:
   *   mean_k -- (NumericMatrix) the bias corrected mean of the forecast
   *             for each member of the ensemble (columns) for each
   *             time period (rows)
   *   y -- (NumericVector) the quantity to be forecast
   *   squared_error -- (arma::mat) A matrix with the squared errors for each
   *                    member of the ensemble (columns) and each time
   *                    period (rows)
   *   priors -- (NumericVector) the prior distribution for the weights (chosen
   *             by the user)
   *   weights -- (NumericVector) the initial weights to use in the algorithm
   *   sigma -- (double) the intial standard deviation for the normal distribution
   *   maxiter -- (int) the maximum number of iterations defaults to 1000
   */

  //The number of time periods
  double T = mean_k.nrow();
  //The number of forecasts in the ensemble
  double K = mean_k.ncol();

  //Initialize the variables
  double log_lik = 0;
  double log_lik_1 = 1;
  NumericVector weights_1(weights.size());
  weights_1 = weights;
  double sigma_1 = 1;
  arma::mat z_1 = arma::zeros<arma::mat>(T,K);

  //Initialize the flag variables for convergence across each iteration
  double lik_diff = 1; double w_diff = 1; double sigma_diff = 1; double z_diff = 1;

  int counter = 1;

  while (lik_diff > 1e-5 || w_diff > 1e-5 || sigma_diff > 1e-5 || z_diff > 1e-5 || counter <= maxiter) {

    //Get the numerator of equation 6 from raftery et al. (2005)
    NumericMatrix znum(T, K);
    if (any(is_na(priors))) {
      //No priors, use the regular weights
      // This is the numerator of Raftery et al (2005) equation 6
      znum = wg_k(mean_k, y, weights, sigma);
    } else {
      //priors given by the user -- estimate the numerator of equation 6
      //from Raftery et al (2005) using these priors
      znum = wg_k(mean_k, y, priors, sigma);
    }

    //Get the denominator by summing each row
    NumericVector zdenom(T);
    for (int i = 0; i < T; ++i) {
      NumericVector temp_k = znum(i, _ );
      zdenom[i] = sum(temp_k);
    }

    // Get Z
    NumericMatrix z(T, K);
    for (int j = 0; j < K; ++j) {
      for (int i = 0; i < T; ++i) {
	double temp_z = znum(i, j) / zdenom[i];
	z(i, j) = temp_z;
      }
    }

    //The log likelihood
    log_lik = sum(log(zdenom));

    //Get the weights
    for (int j = 0; j < K; ++j) {
	NumericVector temp_k = z( _ , j);
	double temp = sum(temp_k) / T;
	weights[j] = temp;
      }

    // -- The standard devation --
    //Convert z into an arma matrix
    arma::mat z_arma = as<arma::mat>(z);
    arma::mat z_squared_error_arma = z_arma % squared_error;
    NumericMatrix z_squared_error = wrap(z_squared_error_arma);
    //Now sum over each row
    NumericVector z_squared_error_sum(T);
    for (int i = 0; i < T; ++i) {
      NumericVector temp_vector = z_squared_error( i, _);
      z_squared_error_sum[i] = sum(temp_vector);
    }
    sigma = (1/T)*sum(z_squared_error_sum);

    //The differences between the previous iterations
    lik_diff = log_lik - log_lik_1;
    w_diff = abs(max(weights - weights_1));
    sigma_diff = abs(sigma - sigma_1);
    arma::mat z_diff_mat = abs(z_arma - z_1);
    z_diff = z_diff_mat.max();

    //Store the current parameters of interest for comparison in the
    //next iteration
    log_lik_1 = log_lik;
    weights_1 = weights;
    sigma_1 = sigma;
    z_1 = z_arma;

    //Update the counter
    counter++;

  } //End of while loop

  //Return an R list with the appropriate information

  List out = List::create( weights, sigma, log_lik);

  return out;

} // End of em() function
