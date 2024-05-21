current_dir=$(pwd)
echo "Current directory: $current_dir"

# Install dependencies
package_dirs=(
    "./rsl_rl"
    "./ee478_utils"
    "./legged_gym"
)

for package_dir in "${package_dirs[@]}"; do
    # Ensure the package exists
    if [ ! -d "$package_dir" ]; then
        echo "Error: Package '$package_dir' does not exist!"
        continue
    fi

    # Go to the target package folder and install
    cd "$package_dir" || exit
    pip install -e .
    
    cd $current_dir
done

