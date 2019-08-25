Feature Engineering done:-

I have extracted "hour" from the 'date' column in the .csv file. I have calculated average output associated by each hour.I have increased its weight by multiplying this group average by 25 and used this "weighted group average" insted of hour. I have added this  column in the matrix to compute gradient descent.




Gradient Descent behaviour:-

Gradient descent is performing approximately same for both p=1 & p=2(p=2 is giving slightly better result). 
I have initially taken Learning Rate (alpha)=0.00000001, which is decreasing dynamically after each update of weight vector , by using formula:-  alpha=alpha/1.0001
so it could converge to minima.

I have repeted gradient descent 10 times for each row of csv file.
I have used value of lamda(rugularization parameter) as 5. 

For best weight vector:-
lamda_reg(rugularization parameter)=5
p=2
Initial Learning Rate (alpha)=0.00000001


Files:-
weight.pickle contain the best weight vector which is returned by get_my_best_weight_vector():

output.csv is the file uploaded in kaggle competetion.


