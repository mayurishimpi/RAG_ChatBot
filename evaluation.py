import evaluate

if __name__ == "__main__":
    
    # Predictions taken from responses geenrated by the model
    prediction= [""""The Nightingale" by Kristin Hannah is a powerful exploration of human resilience. 
                 The main characters, Vianne and Isabelle, are forced to make difficult choices to protect their family during the Nazi occupation of France in World War II. 
                 Despite the harsh circumstances, they demonstrate an incredible capacity for endurance and courage. 
                 The novel delves into the resilience of the human spirit, offering readers a reflection on the human capacity to survive and thrive in challenging circumstances. 
                 The themes of love, loss, and the human condition are timeless and resonate across different times and cultures. The novel received widespread acclaim for its compelling storytelling, well-developed characters, and emotional depth. 
                 It explores the bond between sisters and the strength of women in adversity, presenting a nuanced perspective on femininity and strength. 
                 Both Isabelle and Vianne's characters show significant personal growth and their struggles and triumphs resonate with readers, adding depth to the exploration of the human experience during wartime.""", """The author of the novel 'The Nightingale' is Kristin Hannah.""", """Vianne, one of the main characters in "The Nightingale", is a resilient woman who is forced to share her home with the enemy during the Nazi occupation of France 
                in World War II. She is characterized by her strength and determination to protect her family in the face of adversity. Vianne's experiences throughout the novel highlight
                the resilience of the human spirit and the capacity for endurance and courage in challenging circumstances. Despite the historical setting, her character and the struggles
                she faces resonate universally, addressing timeless themes of love, loss, and the human condition."""]
    
    # References taken from the document: 'The_Nightingale.txt'
    reference = ["""Through Vianne and Isabelle's experiences, the novel delves into the resilience of the human spirit, offering readers a reflection on the capacity for endurance and courage in challenging circumstances.""", """Title: "The Nightingale". A novel by Kristin Hannah""", """Vianne is a resilient character who faces the challenges of Nazi occupation in Carriveau. 
Forced to share her home with the enemy, she makes tough choices to protect her family."""]

    rouge = evaluate.load('rouge')

    # Computing the rouge score
    results = rouge.compute(predictions=prediction, references=reference, use_aggregator=True)
    print(results)