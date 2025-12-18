<img width="1920" height="967" alt="flowfieldout" src="https://github.com/user-attachments/assets/92fba621-1d6b-409a-839b-dcb4d6f3fd91" />

<img width="1920" height="1080" alt="frame2" src="https://github.com/user-attachments/assets/72e54159-bd79-4092-ace0-22ce501dbb41" />

<img width="1920" height="1080" alt="frame1" src="https://github.com/user-attachments/assets/98ac301d-904a-4cc0-8883-7ed7e1cbfbbd" />

In this project, I implemented a dense optical flow algorithm using the Horn-Schunck method to estimate motion between two consecutive frames of a video sequence. Optical flow is a fundamental problem in image processing, where the objective is to infer apparent motion in a scene of two frames, only based on changes in image intensity. The goal of this project was to build the complete optical flow pipeline from first principles, including mathematical formulation, numerical solution, and visualization, rather than just using a pre-built library to implement. 
The approach is based on the assumption that brightness remains constant, which states that the intensity of a moving pixel remains constant across successive frames. This assumption leads to an undetermined optical flow constraint equation, which I resolved using the Horn-Schunck global smoothness framework. By formulating the optical flow estimation as an energy minimization problem, my algorithm enforces spatial smoothness across the image. The resulting Euler-Lagrange equations are solved iteratively to compute horizontal and vertical motion components for every pixel in the frame. 
The system processes video input by extracting two consecutive frames, converting them to grayscale, and applying spatial smoothing to reduce noise. Spatial and temporal intensity gradients are then computed and used within the iterative Horn-Schunck solver. The final output is a dense flow field that visually represents the pixel motion across the image. Experimental results demonstrate that the implementation successfully captures coherent motion patterns in the moving parts of the frame, validating both the mathematical model and the computational approach I used in this project. 

The project was carried out through a structured pipeline designed to transform raw video input into a dense optical flow visualization. Each stage of the pipeline was chosen to align with the assumptions of the Horn–Schunck method while maintaining numerical stability and interpretability of the results. The overall workflow consists of frame extraction, preprocessing, gradient computation, iterative optical flow estimation, and visualization.
The first step involved extracting consecutive frames from a video sequence and converting them to grayscale. Since optical flow relies on intensity variations rather than color information, grayscale conversion reduces computational complexity without loss of relevant motion information. A Gaussian blur was applied to each frame to suppress high-frequency noise, which improves the reliability of spatial and temporal gradient calculations used later in the algorithm.
Following preprocessing, spatial and temporal intensity gradients were computed between the two frames. These gradients represent changes in intensity along the horizontal, vertical, and temporal dimensions and serve as the core inputs to the optical flow constraint equation. Finite-difference approximations were used to estimate these gradients at each pixel location.
The Horn–Schunck optical flow algorithm was then applied to estimate motion vectors across the entire image. Instead of computing motion locally, the Horn–Schunck method formulates optical flow as a global optimization problem that enforces smoothness across neighboring pixels. An iterative numerical solver was used to minimize the associated energy functional, updating the horizontal and vertical flow components at each iteration until convergence or a fixed iteration limit was reached. A regularization parameter was selected to balance sensitivity to motion against smoothness of the resulting flow field.
Finally, the computed optical flow was visualized to evaluate the effectiveness of the implementation. The dense motion field was rendered using vector-based visualizations, allowing for intuitive interpretation of motion direction and magnitude. This visualization step enabled qualitative analysis of motion patterns and helped validate that the algorithm was correctly capturing pixel-level movement within the scene.

There are 7 math steps that we’ll go over. 
Step 1 begins with the brightness constancy assumption, which states that the intensity of a physical point in the scene does not change as it moves between consecutive frames. If a point located at (x,y) at time t moves with velocity components u = dx/dt and v = dy/dt, its image intensity can be written as I(x(t),y(t),t)=C, where C is constant. Differentiating this expression with respect to time and applying the chain rule yields DI/Dx * dx/dt + DI/Dy * dy/dt * DI/Dt = 0. Using standard notation for image gradients, this results in the optical flow constraint equation Ixu+Ixv+It=0, which relates pixel motion to spatial and temporal changes in image intensity and forms the fundamental data constraint for optical flow estimation.

<img width="423" height="459" alt="image" src="https://github.com/user-attachments/assets/f87c933f-5ad2-4d76-88cc-e02ad6d09a84" />

Step 2 is to define the Horn-Schunck Energy functional. We can’t solve for two unknown flow components ( which are u(x,y) and v(x,y) ) at each pixel. We need to introduce additional assumptions to obtain a unique and stable solution. Horn-Schunck revolves this by formulating optical flow as a global optimization problem. Rather than enforcing the optical flow constraint exactly at every pixel, the solution is chosen to approximately satisfy the constraint while also remaining spatially smooth across the image. These two objectives are combined into a single energy functional. Within the integral below, the first term is the smoothness term, which penalizes large spatial variations in the flow field. This term enforces the assumption that neighboring pixels tend to move similarly, producing a coherent and physically plausible motion estimate across the image. The second is the data term, which penalizes violations of the optical flow constraint equation, encouraging the estimated motion field to explain the observed changes in image intensity between frames. 

<img width="467" height="165" alt="image" src="https://github.com/user-attachments/assets/4ba47aa8-0c89-4645-99ed-3a9adca9f104" />

Steps 3 and 4 apply the Euler–Lagrange equations to minimize the Horn–Schunck energy functional. The energy is written as an integral over the image domain of a function L(u,v,ux,uy,vx,vy), which contains both the data fidelity term and the smoothness term. Taking partial derivatives of L with respect to u and its spatial derivatives and substituting them into the Euler–Lagrange equation yields a partial differential equation relating the flow component uuu to the image gradients. The smoothness term results in a Laplacian operator ∇2u, while the data term produces a correction proportional to Ix(Ixu+Ixv+It). An identical process is applied for v resulting in a pair of coupled partial differential equations that describe the optimal optical flow field and form the basis for the iterative Horn–Schunck update equations.

<img width="340" height="455" alt="image" src="https://github.com/user-attachments/assets/f9f9b3aa-4492-499c-af94-28589ee75eca" />


In Step 5, the coupled partial differential equations obtained from the Euler–Lagrange formulation are converted into a numerically solvable form. The Laplacian terms ∇2u and ∇2v are approximated using a finite-difference stencil, replacing the continuous second derivatives with local neighborhood averages of the flow field. Substituting these discrete approximations into the Euler–Lagrange equations yields a pair of linear equations in uuu and vvv at each pixel, which can be expressed in matrix form. Solving this linear system leads to closed-form update expressions for the flow components, where each update consists of a neighborhood-averaged estimate corrected by the optical flow constraint residual. These update equations form the basis of the iterative Horn–Schunck algorithm used in the implementation.

<img width="378" height="485" alt="image" src="https://github.com/user-attachments/assets/d3fe2182-fb5a-4e05-af41-0e626a636b4f" />



Step 6: The linearized update equations for the flow components are solved using an iterative scheme. At each iteration, the horizontal and vertical flow estimates are first replaced by their local neighborhood averages, enforcing the smoothness constraint. A correction term proportional to the optical flow constraint residual Ix​uˉ(k)+Iy​vˉ(k)+It​ is then subtracted, scaled by the image gradients and normalized by α2+Ix2+Iy2​. This process is repeated for a fixed number of iterations or until convergence, progressively refining the flow field toward the minimum of the Horn–Schunck energy functional. 
Step 7: To compute the required image gradients, spatial derivatives of the intensity function are approximated using central finite differences. The horizontal and vertical gradients are defined while the temporal derivative is obtained from intensity differences between consecutive frames. These discrete approximations enable practical computation of the optical flow equations on a pixel grid while maintaining consistency with the continuous mathematical formulation.

<img width="439" height="417" alt="image" src="https://github.com/user-attachments/assets/b1d1f684-8010-4652-a808-503c00656e7d" />


Code Understanding:
Extract_frame.py
The point of this code is just to reach into your folder, grab the mp4, go to the certain frame you specify (I had a 3 frame video, so I chose 0), grab that and the next frame, and save it to the same folder with the names listed.

<img width="609" height="560" alt="image" src="https://github.com/user-attachments/assets/60584c23-640e-4bb6-93c1-59ae42a098f6" />




The spatial image gradients are computed using central finite-difference approximations, where the horizontal and vertical derivatives Ix and Iy are estimated by subtracting neighboring pixel intensities and dividing by two, providing a stable and second-order accurate approximation of the continuous derivatives. Edge padding is applied prior to differencing to preserve image dimensions and avoid introducing artificial boundary artifacts. Smoothness of the optical flow field is enforced through a four-neighbor averaging operator, which replaces each flow value with the mean of its north, south, east, and west neighbors. This averaging operation numerically approximates the Laplacian term in the Horn–Schunck energy functional and encourages spatially coherent motion across the image.

<img width="455" height="286" alt="image" src="https://github.com/user-attachments/assets/b0e3e9a1-1e09-431e-9c34-7bee8fdb86ff" />


This loop implements the fixed-point iterative update of the Horn–Schunck algorithm. At each iteration, the current flow estimates are first replaced by their neighborhood averages to enforce the smoothness constraint. The optical flow constraint residual Ix​uˉ(k)+Iy​vˉ(k)+It​ is then computed and used to correct the smoothed estimates, with the update normalized by α2+Ix2+Iy2 to ensure numerical stability. Repeating this process progressively refines the flow field toward the minimum of the Horn–Schunck energy functional.

<img width="480" height="176" alt="image" src="https://github.com/user-attachments/assets/51c062a9-9097-4c37-b577-08f6c901f080" />


The estimated optical flow field is used to warp the first frame toward the second frame by displacing each pixel according to its computed motion vector. This provides a practical validation of the flow estimate: if the optical flow is accurate, the warped image should closely resemble the second frame. The resulting warped image is saved for visual inspection and later comparison with the original second frame.

<img width="424" height="51" alt="image" src="https://github.com/user-attachments/assets/f977d298-4991-4168-8a44-bac32f7204e5" />



