### Pricing both European Options and American Options with the help of solving the following BSDEs ### 
$$ X_t = X_0 + \int_{t_0}^t b(X_s) ds + \int_{t_0}^t \sigma(X_s) dW_s $$
$$ Y_t = g(X_T) + \int_t^T f_s(X_s,Y_s,Z_s) ds - \int_t^T Z_s dW_s + K_T - K_t $$
$$ Y_t \geq g(X_t), $$
