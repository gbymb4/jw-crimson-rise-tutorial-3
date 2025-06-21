# CNN Architectures Homework - Questions

## Part 1: Implementation Results

1. **Complete the results table:**

| Model Name | Test Accuracy | Training Time (s) | Parameters |
|------------|---------------|-------------------|------------|
| BasicCNN   |               |                   |            |
| ResidualCNN|               |                   |            |
| MultiScaleCNN|             |                   |            |

2. **Which model performed best and why do you think this happened?**
   - Answer: 

3. **Which model had the most parameters? How does this relate to its performance?**
   - Answer: 

## Part 2: Architecture Analysis

4. **In your ResidualBlock implementation, what happens to the gradients during backpropagation that makes training easier?**
   - Hint: The skip connection creates a direct path for gradients to flow backwards. What problem does this solve in deep networks? See: https://medium.com/@amanatulla1606/vanishing-gradient-problem-in-deep-learning-understanding-intuition-and-solutions-da90ef4ecb54
   - Answer: 

5. **For the MultiScaleBlock, why is it important that all three parallel paths output the same number of channels?**
   - Answer: 

6. **What are the computational trade-offs of using parallel convolutions (MultiScaleCNN) vs sequential convolutions (BasicCNN)?**
   - Hint: Parallel paths can run at the same time but what does this mean about memory used?. How does this compare to the BasicCNN?
   - Answer: 

## Part 3: Training Observations

7. **Looking at your training curves, which model showed the most stable training (least fluctuation in loss)?**
   - Answer: 

8. **Did any model show signs of overfitting? How can you tell?**
   - Answer: 

9. **If you had to deploy one of these models in a mobile app with limited computing power, which would you choose and why?**
   - Answer: 

## Part 4: Understanding CNNs

10. **Why are CNNs generally better than fully connected networks for image classification tasks?**
    - Answer: 

11. **How would you expect these three architectures to perform on a more complex dataset like CIFAR-10? Rank them and explain your reasoning.**
    - Answer: 

12. **Suggest one improvement you could make to each architecture:**
    - BasicCNN: 
    - ResidualCNN: 
    - MultiScaleCNN: