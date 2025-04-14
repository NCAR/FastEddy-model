$(document).ready(function() {
    // Add 15 to get to end of URL FastEddy-model/
    var proj_end = document.baseURI.indexOf("FastEddy-model") + 15;
    var end = document.baseURI.indexOf("/", proj_end);
    var cur_ver = document.baseURI.substring(proj_end, end);
    var name = cur_ver.startsWith('v') ? cur_ver.substring(1) : cur_ver;
    var mylist = $("#version-list");
    mylist.empty();
    mylist.append($("<option>", {value: "../" + cur_ver, text: name}));
    $.getJSON(version_json_loc, function(obj) {
        $.each(obj.versions, function() {
            if (this != cur_ver) {
                name = this.startsWith('v') ? this.substring(1) : this;
                mylist.append($("<option>", {value: DOCUMENTATION_OPTIONS.URL_ROOT + '../' + this, text: name}));
            }
        });
    });
});
