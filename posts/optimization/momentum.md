# Gradient Descent

```
new_w = w - ( (lr)(derivative) )
```

# Gradient Descent with Momentum

```
            _______________grad_actual_______________
new_w = w - ( (lr)(derivative) + (p)(grad_anterior) )
```

`p` is usually 0.9



# Nesterov Momentum


```
            __________________grad_actual_________________
new_w = w - ( (lr)(1-p)(derivative) + (p)(grad_anterior) )
```
