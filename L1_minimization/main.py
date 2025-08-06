"""
Where our works begin
"""
from L1_minimization import *
from util import *
from L1 import *
from tqdm import tqdm
import time

def get_random_coordinates(width, height, percentage):
    """
    This function is designed for randomly create coordinate base
    on the shape of image

    It receives the width and height of a Numpy array(image) and
    how many percentage of image need to be missing
    Finally, it returns coordinates of missing value
    """
    image_with_missing_values = []

    max_num = int(percentage * width * height)

    indices = np.random.choice(width * height, size=max_num, replace=False)

    cols_positions = indices // width
    rows_positions = indices % width

    coordinates = np.stack((rows_positions, cols_positions), axis=1)

    image_with_missing_values.append(coordinates)

    return image_with_missing_values

def random_destroy_to_image(image_array : np.ndarray, missing_value : int = 125,
                            percentage : float = 0.1) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    This function is design for creating different cases of destroy to the image.
    (That is, adding different numbers of missing values to the image)

    It receives the image Numpy array, an arbitrary setting missing values and a
    float percentage value to imply how many missing values need to be added.

    It returns a list of Numpy array, which are destroyed image.
    """

    # get the width and length of image
    width, height = image_array.shape

    # generate random coordinates that need to be missing value
    missing_coordinates = get_random_coordinates(width, height, percentage)[0]

    # create a mask to show where are missing values
    mask_image = image_array.copy()

    # replace missing values with arbitrary values
    for x, y in missing_coordinates:
        mask_image[x][y] = missing_value

    return mask_image, missing_coordinates

def get_report_name(t_file_names : bytes) :
    """
    This function is designed for generating report name from
    the CIFAR-10 dataset

    It gets the filename from CIFAR-10 dataset, which is a bytes
    string, converting the bytes into str and then return the final
    names.
    """
    file_name = t_file_names.decode('utf-8')

    name_len = len(file_name)

    report_name = file_name[:name_len - 4] + "_report" + file_name[-4:]

    return report_name

'''def run_batch_recovery(num_images=50):
    cifar_folder = download_and_extract_cifar10("../")
    test_batch_path = os.path.join(cifar_folder, "test_batch")
    test_images, file_names = get_test_image(test_batch_path)

    os.makedirs("../reports", exist_ok=True)

    progress_bar = tqdm(range(num_images), desc="📝 Generating Reports", unit="img")

    for index in progress_bar:
        t_image = test_images[index]
        t_file_name = get_report_name(file_names[index])

        # Convert to grayscale
        original_image = cv.cvtColor(t_image, cv.COLOR_RGB2GRAY)

        # Simulate missing pixels
        destroy_image, missing_coords = random_destroy_to_image(
            original_image, missing_value=0, percentage=0.3)
        missing_set = set(map(tuple, missing_coords))

        # Run recovery
        recovered_image, accuracy = do_image_recovery(destroy_image, missing_set, 0.2)

        if recovered_image is None:
            print(f"⚠️ Skipping image {index + 1} due to recovery failure.")
            continue

        # Generate and save report
        generate_report(
            original_image,
            destroy_image,
            recovered_image,
            t_file_name,
            save_path="../reports"
        )
'''

def run_batch_recovery(num_images=100):
    cifar_folder = download_and_extract_cifar10("../")
    test_batch_path = os.path.join(cifar_folder, "test_batch")
    test_images, file_names = get_test_image(test_batch_path)

    os.makedirs("../reports", exist_ok=True)

    for index in tqdm(range(num_images), desc="📝 Generating Reports", unit="img"):
        t_image = test_images[index]
        t_file_name = get_report_name(file_names[index])

        # Convert to grayscale
        original_image = cv.cvtColor(t_image, cv.COLOR_RGB2GRAY)

        # Simulate missing pixels
        destroy_image, missing_coords = random_destroy_to_image(
            original_image, missing_value=0, percentage=0.3)
        missing_set = set(map(tuple, missing_coords))

        # Run recovery
        recovered_image, accuracy = do_image_recovery(destroy_image, missing_set, 0.2)

        if recovered_image is None:
            print(f"⚠️ Skipping image {index + 1} due to recovery failure.")
            continue

        # Generate and save report
        generate_report(
            original_image,
            destroy_image,
            recovered_image,
            t_file_name,
            save_path="../reports"
        )

def run_resized_image_recovery(image_path, resize_sizes=[64, 128, 256], missing_percentage=0.3):
    """
    Run image recovery on different resized versions of a single image.
    """
    os.makedirs("../resized_reports", exist_ok=True)

    original_image_pil = Image.open(image_path).convert("L")  # Convert to grayscale

    for size in resize_sizes:
        resized_image = original_image_pil.resize((size, size), Image.LANCZOS)
        resized_np = np.array(resized_image)

        # Simulate missing pixels
        destroyed_image, missing_coords = random_destroy_to_image(
            resized_np, missing_value=0, percentage=missing_percentage
        )
        missing_set = set(map(tuple, missing_coords))

        # Measure recovery time
        start_time = time.time()
        recovered_image, accuracy = do_image_recovery(destroyed_image, missing_set, 0.2)
        elapsed_time = time.time() - start_time

        if recovered_image is None:
            print(f"⚠️ Skipping {size}x{size} due to recovery failure.")
            continue

        # Save report
        report_name = f"Lenna_{size}x{size}_report.png"
        generate_report(
            original_image=resized_np,
            masked_image=destroyed_image,
            recovered_image=recovered_image,
            save_file_name=report_name,
            save_path="../resized_reports",
            recovery_time=elapsed_time
        )

        print(f"✅ Report generated for {size}x{size} — Time: {elapsed_time:.2f}s, Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    run_resized_image_recovery(
        image_path="/Users/ehsieh2/Downloads/STEMFall2025/test_images/Lenna_(test_image).png"
    )
# if __name__ == "__main__":
#     import cv2 as cv
#     import time

#     # Load grayscale image
#     original_image = cv.imread("/Users/ehsieh2/Desktop/STEMFall2025/test_images/smallTriangle.png", cv.IMREAD_GRAYSCALE)

#     if original_image is None:
#         raise FileNotFoundError("Image not found. Check the path.")

#     # Add missing values (simulate corruption)
#     destroy_image, missing_coords = random_destroy_to_image(original_image, missing_value=125, percentage=0.3)
#     missing_set = set(map(tuple, missing_coords))

#     # Run your L1 minimization recovery
#     start = time.time()
#     recovered_image, accuracy = do_image_recovery(destroy_image, missing_set, .2)
#     end = time.time()

#     print(f"Recovery completed in {end - start:.4f} seconds.")
#     print(f"Accuracy: {accuracy:.4f}")

#     # Generate and save the report
#     generate_report(
#         original_image,
#         destroy_image,
#         recovered_image,
#         "smallTriangle_report.png"
#     )

# Multiple image test

# if __name__ == "__main__"  :

#     test_images, file_names = get_test_image("../cifar-10-batches-py/test_batch")

#     for index in range(10) :
#         t_image = test_images[index]
#         t_file_name = get_report_name(file_names[index])
#         # read grayscale image
#         original_image = cv.cvtColor(t_image, cv.COLOR_RGB2GRAY)


#         # get image with missing values
#         destroy_image, missing_coords = random_destroy_to_image(original_image, missing_value=0, percentage=0.3)
#         missing_set = set(map(tuple, missing_coords))

#         # Our way to achieve L1 minimization, basically the same
#         # but different in doing Fourier transform

#         start = time.time()
#         recovered_image, accuracy = do_image_recovery(destroy_image, missing_set, .2)
#         end = time.time()
#         print(f"Our total time used: {end - start} seconds.")

#         # Use the L1prunedSTDoptimizer_2D function
#         start_a = time.time()
#         recovered_image_a, accuracy_H_a = L1prunedSTDoptimizer_2D(destroy_image, missing_set, .2)
#         end_a = time.time()
#         print(f"Alex's total time used: {end_a - start_a} seconds.")

#         generate_report(original_image, destroy_image, recovered_image_a,
#                         t_file_name,
#                         save_path="../reports_alex")

#         generate_report(original_image, destroy_image, recovered_image,
#                         t_file_name,
#                         save_path="../reports")


# # Single image test
# if __name__ == "__main__" :
#     # test_images = get_test_image("../cifar-10-batches-py/test_batch")
#
#     # image = cv.cvtColor(test_images[0], cv.COLOR_RGB2GRAY)
#
#     # show(image)
#
#     # read grayscale image
#     original_image = cv.imread("../test_images/smallTriangle.png", cv.IMREAD_GRAYSCALE)
#     #
#     destroy_image, missing_coords = random_destroy_to_image(original_image, missing_value=125, percentage=0.3)
#     missing_set = set(map(tuple, missing_coords))
#
#     # Our way to achieve L1 minimization, basically the same
#     # but different in doing Fourier transform
#     start = time.time()
#     recovered_image, accuracy = do_image_recovery(destroy_image, missing_set, .2)
#     end = time.time()
#     print(f"Our total time used: {end - start} seconds.")
#     print(f"\n H :{recovered_image}; Accuracy: {accuracy}")
#
#     generate_report(original_image, destroy_image, recovered_image,
#                     "smallTriangle_report.png")
#
#     # # Use the L1prunedSTDoptimizer_2D function
#     # start_a = time.time()
#     # recovered_image_a, accuracy_H_a = L1prunedSTDoptimizer_2D(destroy_image, J, .2)
#     # end_a = time.time()
#     # print(f"Alex's total time used: {end_a - start_a} seconds.")
#     #
#     # generate_report(original_image, destroy_image, recovered_image_a,
#                     "smallTriangle_report.png", save_path="../reports_alex")




