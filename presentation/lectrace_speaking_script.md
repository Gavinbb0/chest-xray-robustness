# 10-minute LecTrace presentation script

**Title:** When a chest X-ray gets noisy, does the model still work?

**How to use this script:** Run `lectrace serve presentation/lectrace_presentation.py --port 7001`, then open the local URL that it prints. Press **F** for full screen. Use the right arrow to advance. The numbers below are LecTrace steps, not source-code line numbers. Pause at the indicated steps; move quickly through the others. At a comfortable speaking pace, rehearse once and adjust the pauses to reach ten minutes. Optional figures appear only when copied into `presentation/results/figures/`.

## 0:00-2:30 — Healthcare problem, motivation, and approach (steps 2-5)

Good morning. My project asks a simple question: when the quality of a chest X-ray gets worse, does a pneumonia classifier still make reliable decisions?

This matters because the images seen in practice are not always equally clear. Noise, blur, poor contrast, low brightness, and compression can change how an image looks, even when the patient's underlying condition has not changed. A model that performs well on clean images may behave differently on a degraded image. In this project, a false positive means that a normal image is classified as pneumonia. That could lead to unnecessary follow-up if such a model were ever used in a clinical workflow. A false negative would also be concerning, because pneumonia could be missed. This study examines model behavior; it does not validate a clinical device.

I used the Chest X-Ray Images (Pneumonia) dataset, with Normal and Pneumonia labels. The official test set contains 624 images and was kept separate from training and threshold selection. Both models use the same pretrained ResNet18 architecture and the same basic training settings. The first model was trained using the standard pipeline. The second model was trained with additional image degradations: some images were kept clean, while others were modified with blur, noise, reduced contrast, or reduced brightness. JPEG compression was reserved for evaluation.

During evaluation, I applied the same degradation conditions to both models and compared their predictions and AUROC. I will focus on severe Gaussian noise because it was the most damaging condition for the standard model. Rather than showing every line of code, I will use five execution states to explain how one image changes, how each model responds, and whether that example is consistent with the full test-set result.

## 2:30-3:15 — State 1: the input and ground truth (steps 8-13)

[Pause at the selected case. If the optional case figure is present, point to its top row.]

Here is the selected case, IM-0001-0001. Its ground-truth label is **Normal**. That label is our reference for interpreting the two predictions later. At this stage, the key fact is that we have one chest X-ray and one fixed diagnosis. The picture may be modified for the robustness test, but the patient label is not modified. That distinction matters: if the model changes its answer after the image becomes noisy, the change comes from its response to image quality, not from a change in the target label.

## 3:15-4:15 — State 2: apply noise (steps 17-23)

[Pause where `severity` changes from 0 to 4 and `noise_std` becomes 0.15. Show the optional noise preview if present.]

Now the evaluation moves from the clean condition, Level 0, to Gaussian-noise Level 4. On the image scale from zero to one, the noise standard deviation is 0.15. You can see the visual effect in this preview: fine random variation makes the image harder to read. The ground truth is still Normal. In the actual evaluation, noise was applied before ImageNet normalization, and the same degradation setting was used for both models. Holding the test condition constant makes the comparison more meaningful. The question is whether the model can preserve its distinction between normal and pneumonia despite the added noise.

## 4:15-5:30 — State 3: standard model decision (steps 27-30)

[Show the probability, threshold, and `standard_prediction` in the variable panel.]

The standard model's saved output for this noisy image gives a pneumonia probability of **0.998**. Its decision threshold, selected using the validation set, is **0.8009**. In the LecTrace script, the decision rule compares the probability with that threshold. Since 0.998 is higher, the predicted class is Pneumonia.

But remember that the true label is Normal. This is a high-confidence false positive. What changed is the image quality; the correct label did not change. This state shows a concrete failure mode: severe noise can push a normal image toward an incorrect pneumonia prediction. The probability shown here was produced by the saved experiment. LecTrace is replaying the evaluation logic; it is not running a live ResNet18 forward pass during this presentation.

## 5:30-6:45 — State 4: degradation-aware model decision (steps 33-36)

[Show `robust_probability`, `robust_threshold`, and `robust_prediction`.]

Now we evaluate the same noisy case with the degradation-aware model. Its saved pneumonia probability is **0.194**. Its own validation-derived threshold is **0.5728**. Because 0.194 is below that threshold, this model predicts Normal, which agrees with the ground truth.

The important change between the two models is the training strategy. The architecture and test case are the same, but this model saw varied image quality during training. For this particular example, that training appears to prevent the false pneumonia alarm. This does not mean the second model is always correct. A single image is useful for understanding *how* predictions differ, but we need the full test set to assess whether the improvement is systematic.

## 6:45-8:15 — State 5: from one case to the full test set (steps 39-47)

[Show the AUROC values and, if present, the optional severe-degradation comparison figure.]

The script now reads the saved summary for clean images and Level 4 noise. On clean images, the standard model's AUROC is **0.9219** and the degradation-aware model's is **0.9595**. For the Level 4 noise realization shown in this LecTrace trace, those values become **0.6773** and **0.9541**. So the standard model loses substantial ranking ability under severe noise, while the degradation-aware model changes very little.

AUROC summarizes how well a model ranks pneumonia cases above normal cases across the whole test set. It is different from the threshold decision we saw for one image. The written report also repeats the Gaussian-noise evaluation with three noise realizations. Its Level 4 mean AUROC is **0.6773 plus or minus 0.0133** for the standard model and **0.9559 plus or minus 0.0023** for the degradation-aware model. The robust value in the report differs slightly from the single-realization CSV displayed here because one is a three-seed mean and the other is one saved run. Both support the same narrower conclusion: under the tested synthetic noise, degradation-aware training preserved discrimination much better.

## 8:15-10:00 — Result and healthcare interpretation (steps 49-51)

To conclude, the central result is that clean-image performance alone did not describe what happened when image quality worsened. The standard model performed reasonably well on clean images, but under severe Gaussian noise its AUROC fell sharply. In the selected normal case, it gave a false pneumonia alarm. The degradation-aware model kept its AUROC near its clean level and classified that case correctly.

There is an important clinical interpretation, but also an important boundary. Better robustness to these controlled degradations could make an imaging model less sensitive to routine changes in image quality. However, this experiment used artificially degraded images from one dataset. Four of the five evaluated degradation types were also used in robust training. The models were each trained once, patient-level separation was not explicitly checked, and no external hospital dataset was used. The degradation-aware model also had limited specificity at its fixed test threshold on clean images: **0.5085**. Therefore, high AUROC does not automatically mean good clinical decisions at one threshold, and these results do not establish readiness for clinical use.

My takeaway is this: before trusting a medical image model, we should test how its decisions change when the image changes in ways that leave the diagnosis unchanged. LecTrace helps make that evaluation visible, one meaningful computational state at a time. Thank you.
