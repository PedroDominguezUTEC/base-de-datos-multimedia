const query_image = document.querySelector("#query_image");
var uploaded_image = "";

query_image.addEventListener("change", function(){
    const reader = new FileReader();
    reader.addEventListener("load", () => {
        uploaded_image = reader.result;
        document.querySelector("#preview_upload_image").style.backgroundImage = `url(${uploaded_image})`;
    })
    reader.readAsDataURL(this.files[0]);
})