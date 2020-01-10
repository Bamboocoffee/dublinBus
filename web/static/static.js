function collapse_window() {

    if (content.style.display === "none" && option_container.style.display === "block" && document.getElementById("window").style.width === "400px") {
        document.getElementById("window").style.width = "16px";
        content.style.display = "none";
        option_container.style.display = "none";
        options_open = true;
    } else if (content.style.display === "none" && option_container.style.display === "none" && document.getElementById("window").style.width === "16px" && options_open === true) {
        document.getElementById("window").style.width = "400px";
        option_container.style.display = "block";
        options_open = false;
    } else if (content.style.display === "none") {
        document.getElementById("window").style.width = "400px";
        content.style.display = "block";
    } else {
        document.getElementById("window").style.width = "16px";
        content.style.display = "none";
    }
}


function options_container() {
    if (content.style.display === "none" && form_fill === false) {
        $('#window').css('height', '320px');
        option_container.style.display = "none";
        content.style.display = "block";
    } else if (content.style.display === "none") {
        option_container.style.display = "none";
        content.style.display = "block";
    } else {
        $('#window').css('height', '650px');
        option_container.style.display = "block";
        content.style.display = "none";

    }
}

function hide_window() {
  if ($('#window').css('display', 'block')) {
    $('#window').css('display', 'none');
    window_hidden = true;
  }
};

function show_window() {
  $('#window').css('display', 'block');
};

function check_modal() {
  if(($('#window').css('display', 'none'))) {
    if(($.modal.isActive()) === false) {
      $('#window').css('display', 'block');
    }
  }
};
